# Azure Container Instances Deployment
# Ejecutar estos comandos en Azure CLI

# 1. Login a Azure
az login

# 2. Crear Resource Group (si no existe)
az group create --name CLIP-Container-RG --location "East US"

# 3. Crear Azure Container Registry
az acr create --resource-group CLIP-Container-RG --name clipregistryunique --sku Basic --admin-enabled true

# 4. Build y push de imagen
az acr build --registry clipregistryunique --image clip-comparador:latest .

# 5. Obtener credenciales del registry
$ACR_LOGIN_SERVER = az acr show --name clipregistryunique --resource-group CLIP-Container-RG --query "loginServer" --output tsv
$ACR_USERNAME = az acr credential show --name clipregistryunique --resource-group CLIP-Container-RG --query "username" --output tsv
$ACR_PASSWORD = az acr credential show --name clipregistryunique --resource-group CLIP-Container-RG --query "passwords[0].value" --output tsv

# 6. Crear Container Instance
az container create `
  --resource-group CLIP-Container-RG `
  --name clip-comparador-container `
  --image "$ACR_LOGIN_SERVER/clip-comparador:latest" `
  --registry-login-server $ACR_LOGIN_SERVER `
  --registry-username $ACR_USERNAME `
  --registry-password $ACR_PASSWORD `
  --dns-name-label clipvisualsearch `
  --ports 8000 `
  --memory 4 `
  --cpu 2 `
  --environment-variables PORT=8000

# 7. Obtener URL de acceso
Write-Host "URL de acceso: http://clipvisualsearch.eastus.azurecontainer.io:8000"