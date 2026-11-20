#!/usr/bin/env bash
# One-time preparation of a Google Cloud project for this service: APIs, local
# credentials, a least-privilege service account and a smoke test of the models.
#
# Usage: PROJECT_ID=my-project scripts/setup_gcp.sh
#   REGION defaults to us-central1 for the compute settings; Gemini 3.x itself is served
#   from the `global` location (see .env.example).
set -euo pipefail

: "${PROJECT_ID:?set PROJECT_ID to the GCP project id}"
export REGION="${REGION:-us-central1}"
export SA_NAME="${SA_NAME:-getnet-agents}"
export SA_EMAIL="${SA_NAME}@${PROJECT_ID}.iam.gserviceaccount.com"

echo "== 1. Session and project"
gcloud auth login --brief
gcloud config set project "$PROJECT_ID"
gcloud config set compute/region "$REGION"

echo "== 2. Billing must be linked (this is where the credits live)"
gcloud billing projects describe "$PROJECT_ID" --format="value(billingEnabled,billingAccountName)"
# If it prints False, link the billing account that holds the credits:
#   gcloud billing accounts list
#   gcloud billing projects link "$PROJECT_ID" --billing-account=XXXXXX-XXXXXX-XXXXXX

echo "== 3. Enable APIs"
gcloud services enable \
  aiplatform.googleapis.com \
  cloudresourcemanager.googleapis.com \
  iam.googleapis.com \
  artifactregistry.googleapis.com \
  run.googleapis.com \
  cloudbuild.googleapis.com \
  logging.googleapis.com

echo "== 4. Local credentials for development (ADC)"
gcloud auth application-default login
gcloud auth application-default set-quota-project "$PROJECT_ID"

echo "== 5. Service account for the container (least privilege)"
gcloud iam service-accounts create "$SA_NAME" \
  --display-name="Getnet multi-agent service" 2>/dev/null || echo "   (already exists)"
gcloud projects add-iam-policy-binding "$PROJECT_ID" \
  --member="serviceAccount:${SA_EMAIL}" --role="roles/aiplatform.user" --quiet >/dev/null
gcloud projects add-iam-policy-binding "$PROJECT_ID" \
  --member="serviceAccount:${SA_EMAIL}" --role="roles/logging.logWriter" --quiet >/dev/null
# Local key for docker compose only. Never commit it (.secrets/ is ignored).
mkdir -p .secrets
[ -f .secrets/sa-key.json ] || gcloud iam service-accounts keys create .secrets/sa-key.json --iam-account="$SA_EMAIL"

echo "== 6. Smoke test: chat model and embeddings on Vertex AI"
export GOOGLE_GENAI_USE_VERTEXAI=true
export GOOGLE_CLOUD_PROJECT="$PROJECT_ID"
export GOOGLE_CLOUD_LOCATION="global"
python3 -m venv .venv-setup && .venv-setup/bin/pip install -q -U google-genai
.venv-setup/bin/python - <<'EOF'
from google import genai

client = genai.Client()
reply = client.models.generate_content(model="gemini-3.8-flash", contents="Reply with: OK Getnet")
print("LLM:", reply.text.strip())
vector = client.models.embed_content(model="gemini-embedding-2", contents="card machine")
print("Embedding dimensions:", len(vector.embeddings[0].values))
EOF
rm -rf .venv-setup

echo "== Done. Next: cp .env.example .env, set GOOGLE_CLOUD_PROJECT=$PROJECT_ID, OPERATOR_API_KEY and LOG_SALT, then make install."
