$scriptDir = Split-Path -Parent -Path $MyInvocation.MyCommand.Definition
$env:PYTHONPATH = "$scriptDir;$env:PYTHONPATH"
python -m convert_to_precomputed $args
