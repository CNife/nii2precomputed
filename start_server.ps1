$miniserve = Start-Process `
    -FilePath 'miniserve' `
    -ArgumentList '--verbose', '--header', '"Access-Control-Allow-Origin: *"', '"D:\WorkData"' `
    -PassThru `
    -NoNewWindow
$neuroglancer = Start-Process `
    -FilePath 'pnpm' `
    -ArgumentList 'run', 'dev-server', '--port', '2101' `
    -WorkingDirectory 'C:\Code\neuroglancer' `
    -PassThru `
    -NoNewWindow
$miniserve, $neuroglancer | Wait-Process
