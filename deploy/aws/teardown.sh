#!/usr/bin/env bash
# Remove everything deploy.sh created (stops all charges). Asks before deleting.
#   REGION=eu-west-2 ./deploy/aws/teardown.sh
# Keeps the shared, standard-named roles ecsTaskExecutionRole and
# ecsInfrastructureRoleForExpressServices (other ECS apps may use them; they cost nothing).
set -uo pipefail
export MSYS_NO_PATHCONV=1
REGION="${REGION:-eu-west-2}"
APP="${APP:-gb-demand}"
export AWS_REGION="$REGION" AWS_DEFAULT_REGION="$REGION" AWS_PAGER=""
ACCOUNT="$(aws sts get-caller-identity --query Account --output text)"
BUCKET="${BUCKET:-$APP-artifacts-$ACCOUNT-$REGION}"

echo "This permanently deletes in $REGION: services $APP-api/$APP-dashboard, schedule,"
echo "task definitions, ECR repo $APP (all images), S3 bucket $BUCKET (all data), roles $APP-*."
read -r -p "Type the app name ($APP) to confirm: " answer
[ "$answer" = "$APP" ] || { echo "Aborted."; exit 1; }

aws scheduler delete-schedule --name "$APP-daily-retrain" 2>/dev/null && echo "schedule deleted"
for svc in "$APP-api" "$APP-dashboard"; do
  arn="$(aws ecs describe-services --cluster default --services "$svc" \
         --query 'services[0].serviceArn' --output text 2>/dev/null)"
  if [ -n "$arn" ] && [ "$arn" != "None" ]; then
    aws ecs delete-express-gateway-service --service-arn "$arn" >/dev/null && echo "$svc deleting"
  fi
done
for td in $(aws ecs list-task-definitions --family-prefix "$APP-pipeline" --query 'taskDefinitionArns[]' --output text); do
  aws ecs deregister-task-definition --task-definition "$td" >/dev/null
done
aws logs delete-log-group --log-group-name "/ecs/$APP-pipeline" 2>/dev/null
aws ecr delete-repository --repository-name "$APP" --force >/dev/null 2>&1 && echo "ECR repo deleted"
aws s3 rb "s3://$BUCKET" --force >/dev/null 2>&1 && echo "bucket deleted"
for role in "$APP-task" "$APP-scheduler"; do
  for p in $(aws iam list-role-policies --role-name "$role" --query 'PolicyNames[]' --output text 2>/dev/null); do
    aws iam delete-role-policy --role-name "$role" --policy-name "$p"
  done
  aws iam delete-role --role-name "$role" 2>/dev/null && echo "role $role deleted"
done
echo "Done. Express Mode removes its load balancer as the services finish deleting."
