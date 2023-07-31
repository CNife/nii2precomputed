$dateTag = Get-Date -Format 'yyyyMMdd-HHmmss'
docker build `
    --tag cnife/nii2precomputed:latest `
    --tag cnife/nii2precomputed:$dateTag `
    --push `
    $PSScriptRoot