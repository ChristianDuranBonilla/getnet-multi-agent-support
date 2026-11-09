#!/usr/bin/env bash
# Preparación del proyecto GCP para el reto Getnet. Correr en tu máquina, una sola vez.
# No cuenta como tiempo de desarrollo del core.
set -euo pipefail

export PROJECT_ID="ai-hardcore-engineer-getnet"
export PROJECT_NUMBER="899648446618"
export REGION="us-central1"
export SA_NAME="getnet-agents"
export SA_EMAIL="${SA_NAME}@${PROJECT_ID}.iam.gserviceaccount.com"

echo "== 1. Sesión y proyecto"
gcloud auth login --brief
gcloud config set project "$PROJECT_ID"
gcloud config set compute/region "$REGION"

echo "== 2. Verificar que la facturación esté vinculada (aquí viven tus créditos)"
gcloud billing projects describe "$PROJECT_ID" --format="value(billingEnabled,billingAccountName)"
# Si imprime False, vincula la cuenta de facturación con créditos:
#   gcloud billing accounts list
#   gcloud billing projects link "$PROJECT_ID" --billing-account=XXXXXX-XXXXXX-XXXXXX

echo "== 3. Habilitar APIs"
gcloud services enable \
  aiplatform.googleapis.com \
  cloudresourcemanager.googleapis.com \
  iam.googleapis.com \
  artifactregistry.googleapis.com \
  run.googleapis.com \
  cloudbuild.googleapis.com \
  logging.googleapis.com

echo "== 4. Credenciales locales para desarrollo (ADC)"
gcloud auth application-default login
gcloud auth application-default set-quota-project "$PROJECT_ID"

echo "== 5. Service account para el contenedor (mínimo privilegio)"
gcloud iam service-accounts create "$SA_NAME" \
  --display-name="Getnet multi-agent service" 2>/dev/null || echo "   (ya existe)"
gcloud projects add-iam-policy-binding "$PROJECT_ID" \
  --member="serviceAccount:${SA_EMAIL}" --role="roles/aiplatform.user" --quiet >/dev/null
gcloud projects add-iam-policy-binding "$PROJECT_ID" \
  --member="serviceAccount:${SA_EMAIL}" --role="roles/logging.logWriter" --quiet >/dev/null
# Llave local solo para docker-compose en tu máquina. NO la subas al repo (va en .gitignore).
mkdir -p .secrets
[ -f .secrets/sa-key.json ] || gcloud iam service-accounts keys create .secrets/sa-key.json --iam-account="$SA_EMAIL"

echo "== 6. Entorno Python de prueba"
python3 -m venv .venv && source .venv/bin/activate
pip install -q -U google-genai langchain-google-genai

echo "== 7. Prueba real: modelo + embeddings (si esto responde, los créditos ya se están usando)"
export GOOGLE_GENAI_USE_VERTEXAI=true
export GOOGLE_CLOUD_PROJECT="$PROJECT_ID"
export GOOGLE_CLOUD_LOCATION="$REGION"
python3 - <<'EOF'
from google import genai
c = genai.Client()  # toma Vertex + proyecto + región de las variables de entorno
r = c.models.generate_content(model="gemini-3.8-flash", contents="Responde solo: OK Getnet")
print("LLM:", r.text.strip())
e = c.models.embed_content(model="gemini-embedding-2", contents="máquina de cartão Get Smart")
print("Embedding dims:", len(e.embeddings[0].values))
EOF

echo "== 8. Presupuesto de seguridad (alerta al 50/90% de 2,000 MXN)"
BILLING=$(gcloud billing projects describe "$PROJECT_ID" --format="value(billingAccountName)")
gcloud billing budgets create --billing-account="${BILLING#billingAccounts/}" \
  --display-name="getnet-challenge" --budget-amount=2000MXN \
  --threshold-rule=percent=0.5 --threshold-rule=percent=0.9 \
  --filter-projects="projects/${PROJECT_NUMBER}" 2>/dev/null || echo "   (presupuesto ya existe o sin permiso; opcional)"

echo
echo "Listo. Variables para el proyecto (.env):"
cat <<EOF
GOOGLE_GENAI_USE_VERTEXAI=true
GOOGLE_CLOUD_PROJECT=${PROJECT_ID}
GOOGLE_CLOUD_LOCATION=${REGION}
GOOGLE_APPLICATION_CREDENTIALS=/run/secrets/sa-key.json   # dentro del contenedor
LLM_MODEL=gemini-3.8-flash
EMBEDDING_MODEL=gemini-embedding-2
EOF
