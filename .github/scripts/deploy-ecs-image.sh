#!/usr/bin/env bash
set -euo pipefail

if [[ "$#" -ne 7 ]]; then
  echo "usage: $0 <cluster> <service> <family> <container> <repository> <sha> <region>" >&2
  exit 2
fi

cluster="$1"
service="$2"
family="$3"
container="$4"
repository="$5"
image_tag="$6"
region="$7"

if [[ ! "${image_tag}" =~ ^[0-9a-f]{40}$ ]]; then
  echo "invalid immutable image tag: ${image_tag}" >&2
  exit 2
fi

image_digest="$(aws ecr describe-images \
  --repository-name "${repository}" \
  --image-ids imageTag="${image_tag}" \
  --region "${region}" \
  --query 'imageDetails[0].imageDigest' \
  --output text)"

if [[ ! "${image_digest}" =~ ^sha256:[0-9a-f]{64}$ ]]; then
  echo "ECR image not found or returned an invalid digest: ${repository}:${image_tag}" >&2
  exit 1
fi

account_id="$(aws sts get-caller-identity --query Account --output text)"
image_uri="${account_id}.dkr.ecr.${region}.amazonaws.com/${repository}@${image_digest}"

work_dir="$(mktemp -d)"
trap 'rm -rf "${work_dir}"' EXIT

aws ecs describe-task-definition \
  --task-definition "${family}" \
  --region "${region}" \
  --query taskDefinition \
  --output json >"${work_dir}/current-task-definition.json"

jq --arg container "${container}" --arg image "${image_uri}" '
  if ([.containerDefinitions[] | select(.name == $container)] | length) != 1 then
    error("expected exactly one container named " + $container)
  else
    .
  end
  | .containerDefinitions |= map(
      if .name == $container then .image = $image else . end
    )
  | del(
      .taskDefinitionArn,
      .revision,
      .status,
      .requiresAttributes,
      .compatibilities,
      .registeredAt,
      .registeredBy,
      .deregisteredAt
    )
' "${work_dir}/current-task-definition.json" >"${work_dir}/next-task-definition.json"

task_definition_arn="$(aws ecs register-task-definition \
  --cli-input-json "file://${work_dir}/next-task-definition.json" \
  --region "${region}" \
  --query taskDefinition.taskDefinitionArn \
  --output text)"

aws ecs update-service \
  --cluster "${cluster}" \
  --service "${service}" \
  --task-definition "${task_definition_arn}" \
  --region "${region}" \
  --query 'service.{serviceName:serviceName,taskDefinition:taskDefinition}'

if ! aws ecs wait services-stable \
  --cluster "${cluster}" \
  --services "${service}" \
  --region "${region}"; then
  aws ecs describe-services \
    --cluster "${cluster}" \
    --services "${service}" \
    --region "${region}" \
    --query 'services[0].{status:status,desiredCount:desiredCount,runningCount:runningCount,pendingCount:pendingCount,deployments:deployments,events:events[0:10]}' \
    --output json >&2
  exit 1
fi

echo "deployed ${image_uri} as ${task_definition_arn}"
