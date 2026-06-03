$TaskName   = "AgriDistrictFetch"
$ProjectRoot = "D:\CDAC\Indian-Farmer-Crop-Recommendation-System\agri_crop_recommendation"
$ScriptDir   = Join-Path $ProjectRoot "scripts"
$FetchScript = Join-Path $ScriptDir "fetch_missing_districts.py"
$VenvPython  = Join-Path $ProjectRoot "venv\Scripts\python.exe"
$PythonExe   = if (Test-Path $VenvPython) { $VenvPython } else { "python" }

$Action = New-ScheduledTaskAction `
    -Execute $PythonExe `
    -Argument $FetchScript `
    -WorkingDirectory $ProjectRoot

$Trigger = New-ScheduledTaskTrigger -Daily -At "02:30"

$Settings = New-ScheduledTaskSettingsSet `
    -ExecutionTimeLimit (New-TimeSpan -Hours 6) `
    -RestartCount 2 `
    -RestartInterval (New-TimeSpan -Minutes 30) `
    -StartWhenAvailable `
    -RunOnlyIfNetworkAvailable:$true

$Principal = New-ScheduledTaskPrincipal `
    -UserId "$env:USERDOMAIN\$env:USERNAME" `
    -LogonType S4U `
    -RunLevel Highest

Register-ScheduledTask `
    -TaskName $TaskName `
    -Action $Action `
    -Trigger $Trigger `
    -Settings $Settings `
    -Principal $Principal `
    -Description "Daily Open-Meteo district weather fetch for Indian Farmer Crop Recommendation System" `
    -Force | Out-Null

Write-Host ""
Write-Host "============================================================"
Write-Host "  Scheduled Task Registered Successfully"
Write-Host "============================================================"
Write-Host "  Task name : $TaskName"
Write-Host "  Runs at   : 02:30 AM daily"
Write-Host "  Python    : $PythonExe"
Write-Host "  Script    : $FetchScript"
Write-Host ""
Get-ScheduledTask -TaskName $TaskName | Select-Object TaskName, State
Write-Host ""
Write-Host "  To remove : Unregister-ScheduledTask -TaskName '$TaskName' -Confirm:$false"
Write-Host "  To run now: Start-ScheduledTask -TaskName '$TaskName'"
Write-Host "============================================================"
