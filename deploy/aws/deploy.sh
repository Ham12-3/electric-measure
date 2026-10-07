#!/usr/bin/env bash
# Deploy the forecast API + dashboard to AWS, with a daily retraining job.
#
#   ECR (image) -> ECS Express Mode services (API, dashboard; HTTPS URLs + load balancer)
#   EventBridge Scheduler (daily 06:00 UK) -> ECS Fargate task (pipeline) -> S3 (models, data)
#
# Prerequisites: AWS CLI v2 (recent; Express Mode commands), Docker running, `aws configure`
# done, a default VPC in the region, and local data/models (run the pipeline once locally).
#
# Usage (Git Bash / Linux / macOS):
#   REGION=eu-west-2 ./deploy/aws/deploy.sh
# Re-running is safe: existing resources are reused and services are redeployed.
# Remove everything with ./deploy/aws/teardown.sh
#
# COST: two always-on Fargate services + a load balancer, roughly $90-110/month at the
# default 1 vCPU / 2 GB each. Tear down when you're not demoing it.
set -euo pipefail
export MSYS_NO_PATHCONV=1   # stop Git Bash rewriting "/health" etc. into Windows paths

REGION="${REGION:-eu-west-2}"
APP="${APP:-gb-demand}"
export AWS_REGION="$REGION" AWS_DEFAULT_REGION="$REGION" AWS_PAGER=""
ROOT="$(cd "$(dirname "$0")/../.." && pwd)"

step() { printf '\n==> %s\n' "$*"; }

ACCOUNT="$(aws sts get-caller-identity --query Account --output text)"
BUCKET="${BUCKET:-$APP-artifacts-$ACCOUNT-$REGION}"
REGISTRY="$ACCOUNT.dkr.ecr.$REGION.amazonaws.com"
IMAGE="$REGISTRY/$APP:latest"
API_URL="https://$APP-api.ecs.$REGION.on.aws"
DASH_URL="https://$APP-dashboard.ecs.$REGION.on.aws"
echo "account=$ACCOUNT region=$REGION bucket=$BUCKET"

# --------------------------------------------------------------------------- storage
step "S3 bucket (private) for models and data"
if ! aws s3api head-bucket --bucket "$BUCKET" 2>/dev/null; then
  if [ "$REGION" = "us-east-1" ]; then
    aws s3api create-bucket --bucket "$BUCKET" >/dev/null
  else
    aws s3api create-bucket --bucket "$BUCKET" \
      --create-bucket-configuration "LocationConstraint=$REGION" >/dev/null
  fi
  aws s3api put-public-access-block --bucket "$BUCKET" --public-access-block-configuration \
    BlockPublicAcls=true,IgnorePublicAcls=true,BlockPublicPolicy=true,RestrictPublicBuckets=true
fi
step "Seed S3 with local models and data (so the API works before the first retrain)"
for d in artifacts/models data/processed data/raw; do
  [ -d "$ROOT/$d" ] && aws s3 sync "$ROOT/$d" "s3://$BUCKET/$d" --only-show-errors --exclude "*.tmp"
done
aws s3 ls "s3://$BUCKET/artifacts/models/meta.json" >/dev/null \
  || { echo "No models in S3. Run 'python -m demand_forecast.serving.production' first."; exit 1; }

# --------------------------------------------------------------------------- image
step "ECR repository + image push"
aws ecr describe-repositories --repository-names "$APP" >/dev/null 2>&1 \
  || aws ecr create-repository --repository-name "$APP" \
       --image-scanning-configuration scanOnPush=true >/dev/null
aws ecr get-login-password | docker login --username AWS --password-stdin "$REGISTRY"
docker build --platform linux/amd64 -t "$IMAGE" "$ROOT"
docker push "$IMAGE"

# --------------------------------------------------------------------------- IAM
trust() { printf '{"Version":"2012-10-17","Statement":[{"Effect":"Allow","Principal":{"Service":"%s"},"Action":"sts:AssumeRole"}]}' "$1"; }
ensure_role() {  # name, service principal
  aws iam get-role --role-name "$1" >/dev/null 2>&1 \
    || aws iam create-role --role-name "$1" --assume-role-policy-document "$(trust "$2")" >/dev/null
}
step "IAM roles"
ensure_role ecsTaskExecutionRole ecs-tasks.amazonaws.com
aws iam attach-role-policy --role-name ecsTaskExecutionRole \
  --policy-arn arn:aws:iam::aws:policy/service-role/AmazonECSTaskExecutionRolePolicy
ensure_role ecsInfrastructureRoleForExpressServices ecs.amazonaws.com
aws iam attach-role-policy --role-name ecsInfrastructureRoleForExpressServices \
  --policy-arn arn:aws:iam::aws:policy/service-role/AmazonECSInfrastructureRoleforExpressGatewayServices
# App containers: read/write only this project's bucket.
ensure_role "$APP-task" ecs-tasks.amazonaws.com
aws iam put-role-policy --role-name "$APP-task" --policy-name s3-artifacts --policy-document "{
  \"Version\":\"2012-10-17\",\"Statement\":[
    {\"Effect\":\"Allow\",\"Action\":\"s3:ListBucket\",\"Resource\":\"arn:aws:s3:::$BUCKET\"},
    {\"Effect\":\"Allow\",\"Action\":[\"s3:GetObject\",\"s3:PutObject\"],\"Resource\":\"arn:aws:s3:::$BUCKET/*\"}]}"
ensure_role "$APP-scheduler" scheduler.amazonaws.com
EXEC_ROLE="arn:aws:iam::$ACCOUNT:role/ecsTaskExecutionRole"
INFRA_ROLE="arn:aws:iam::$ACCOUNT:role/ecsInfrastructureRoleForExpressServices"
TASK_ROLE="arn:aws:iam::$ACCOUNT:role/$APP-task"
SCHED_ROLE="arn:aws:iam::$ACCOUNT:role/$APP-scheduler"
aws iam put-role-policy --role-name "$APP-scheduler" --policy-name run-pipeline --policy-document "{
  \"Version\":\"2012-10-17\",\"Statement\":[
    {\"Effect\":\"Allow\",\"Action\":\"ecs:RunTask\",\"Resource\":\"arn:aws:ecs:$REGION:$ACCOUNT:task-definition/$APP-pipeline:*\"},
    {\"Effect\":\"Allow\",\"Action\":\"iam:PassRole\",\"Resource\":[\"$EXEC_ROLE\",\"$TASK_ROLE\"]}]}"
echo "waiting 20s for IAM changes to propagate..."; sleep 20

# --------------------------------------------------------------------------- services
aws ecs create-cluster --cluster-name default >/dev/null
service_active() {
  [ "$(aws ecs describe-services --cluster default --services "$1" \
        --query 'services[0].status' --output text 2>/dev/null)" = "ACTIVE" ]
}
express_service() {  # name, port, health path, primary-container JSON
  if service_active "$1"; then
    echo "$1 exists - redeploying with the new image"
    aws ecs update-service --cluster default --service "$1" --force-new-deployment >/dev/null
  else
    aws ecs create-express-gateway-service --service-name "$1" \
      --execution-role-arn "$EXEC_ROLE" --infrastructure-role-arn "$INFRA_ROLE" \
      --task-role-arn "$TASK_ROLE" --primary-container "$4" --health-check-path "$3" \
      --scaling-target '{"minTaskCount":1,"maxTaskCount":2}' >/dev/null
  fi
}
step "API service (ECS Express Mode)"
express_service "$APP-api" 8000 /health \
  "{\"image\":\"$IMAGE\",\"containerPort\":8000,\"environment\":[{\"name\":\"ARTIFACT_BUCKET\",\"value\":\"$BUCKET\"}]}"
step "Dashboard service (ECS Express Mode)"
express_service "$APP-dashboard" 8501 /_stcore/health \
  "{\"image\":\"$IMAGE\",\"containerPort\":8501,\"command\":[\"streamlit\",\"run\",\"dashboard/app.py\",\"--server.address=0.0.0.0\",\"--server.port=8501\",\"--server.headless=true\"],\"environment\":[{\"name\":\"API_URL\",\"value\":\"$API_URL\"}]}"

# --------------------------------------------------------------------------- daily retrain
step "Pipeline task definition (Fargate, 1 vCPU / 4 GB)"
aws logs create-log-group --log-group-name "/ecs/$APP-pipeline" 2>/dev/null || true
TASKDEF="$(aws ecs register-task-definition --family "$APP-pipeline" \
  --requires-compatibilities FARGATE --network-mode awsvpc --cpu 1024 --memory 4096 \
  --execution-role-arn "$EXEC_ROLE" --task-role-arn "$TASK_ROLE" \
  --container-definitions "[{\"name\":\"pipeline\",\"image\":\"$IMAGE\",\"essential\":true,
    \"command\":[\"python\",\"-m\",\"demand_forecast.serving.pipeline\"],
    \"environment\":[{\"name\":\"ARTIFACT_BUCKET\",\"value\":\"$BUCKET\"}],
    \"logConfiguration\":{\"logDriver\":\"awslogs\",\"options\":{
      \"awslogs-group\":\"/ecs/$APP-pipeline\",\"awslogs-region\":\"$REGION\",\"awslogs-stream-prefix\":\"pipeline\"}}}]" \
  --query taskDefinition.taskDefinitionArn --output text)"
SUBNETS="$(aws ec2 describe-subnets --filters Name=default-for-az,Values=true \
  --query 'Subnets[].SubnetId' --output json | tr -d ' \n\r')"
[ "$SUBNETS" != "[]" ] || { echo "No default-VPC subnets in $REGION; create a default VPC first."; exit 1; }

step "Daily schedule 06:00 UK time (EventBridge Scheduler)"
TARGET="{\"Arn\":\"arn:aws:ecs:$REGION:$ACCOUNT:cluster/default\",\"RoleArn\":\"$SCHED_ROLE\",
  \"EcsParameters\":{\"TaskDefinitionArn\":\"$TASKDEF\",\"LaunchType\":\"FARGATE\",
  \"NetworkConfiguration\":{\"awsvpcConfiguration\":{\"Subnets\":$SUBNETS,\"AssignPublicIp\":\"ENABLED\"}}}}"
SCHEDULE_ARGS=(--name "$APP-daily-retrain" --schedule-expression "cron(0 6 * * ? *)"
  --schedule-expression-timezone Europe/London --flexible-time-window Mode=OFF --target "$TARGET")
if aws scheduler get-schedule --name "$APP-daily-retrain" >/dev/null 2>&1; then
  aws scheduler update-schedule "${SCHEDULE_ARGS[@]}" >/dev/null
else
  aws scheduler create-schedule "${SCHEDULE_ARGS[@]}" >/dev/null
fi

step "Done"
cat <<EOF
API:        $API_URL/docs      (try $API_URL/forecast)
Dashboard:  $DASH_URL
Bucket:     s3://$BUCKET
Retrain:    daily 06:00 Europe/London (logs: CloudWatch /ecs/$APP-pipeline)

Services take a few minutes to become healthy. Check progress with:
  aws ecs describe-services --cluster default --services $APP-api $APP-dashboard \\
    --query 'services[].[serviceName,status,runningCount]' --output table
Run a retrain now:
  aws ecs run-task --cluster default --launch-type FARGATE --task-definition $TASKDEF \\
    --network-configuration 'awsvpcConfiguration={subnets=$(echo "$SUBNETS" | tr -d '[]"'),assignPublicIp=ENABLED}'
EOF
