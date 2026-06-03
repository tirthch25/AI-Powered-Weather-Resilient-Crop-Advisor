# ============================================================
#  Auto-Fetch Scheduler — Indian Farmer Crop Recommendation
# ============================================================
#  Registers a Windows Scheduled Task that runs
#  fetch_missing_districts.py every day at a chosen time.
#
#  The script safely skips already-downloaded district/year
#  files, so it can run repeatedly without duplication.
#
#  Usage (run once from an elevated PowerShell prompt):
#      cd agri_crop_recommendation
#      powershell -ExecutionPolicy Bypass -File scripts\auto_fetch_scheduler.ps1
#
#  To remove the task later:
#      Unregister-ScheduledTask -TaskName "AgriDistrictFetch" -Confirm:$false
# ============================================================

param(
    [string]$RunTime = "02:30",           # Daily trigger time (24h HH:MM)
    [string]$TaskName = "AgriDistrictFetch",
    [switch]$Remove                        # Pass -Remove to delete the task
)

# ── Remove task if requested ──────────────────────────────────────────────────
if ($Remove) {
    Unregister-ScheduledTask -TaskName $TaskName -Confirm:$false -ErrorAction SilentlyContinue
    Write-Host "✅ Scheduled task '$TaskName' removed." -ForegroundColor Green
    exit 0
}

# ── Resolve paths ─────────────────────────────────────────────────────────────
$ScriptDir   = Split-Path -Parent $MyInvocation.MyCommand.Definition
$ProjectRoot = Split-Path -Parent $ScriptDir
$FetchScript = Join-Path $ScriptDir "fetch_missing_districts.py"

# Find python in the project's venv, or fall back to system python
$VenvPython  = Join-Path $ProjectRoot "venv\Scripts\python.exe"
$PythonExe   = if (Test-Path $VenvPython) { $VenvPython } else { "python" }

if (-not (Test-Path $FetchScript)) {
    Write-Error "fetch_missing_districts.py not found at: $FetchScript"
    exit 1
}

Write-Host ""
Write-Host "============================================================" -ForegroundColor Cyan
Write-Host "  Registering Daily District Fetch Task" -ForegroundColor Cyan
Write-Host "============================================================" -ForegroundColor Cyan
Write-Host "  Task name   : $TaskName"
Write-Host "  Run time    : $RunTime daily"
Write-Host "  Python      : $PythonExe"
Write-Host "  Script      : $FetchScript"
Write-Host "  Working dir : $ProjectRoot"
Write-Host ""

# ── Build scheduled task ─────────────────────────────────────────────────────
$Action = New-ScheduledTaskAction `
    -Execute $PythonExe `
    -Argument "`"$FetchScript`"" `
    -WorkingDirectory $ProjectRoot

$Trigger = New-ScheduledTaskTrigger -Daily -At $RunTime

$Settings = New-ScheduledTaskSettingsSet `
    -ExecutionTimeLimit (New-TimeSpan -Hours 6) `
    -RestartCount 2 `
    -RestartInterval (New-TimeSpan -Minutes 30) `
    -StartWhenAvailable `
    -WakeToRun:$false `
    -RunOnlyIfNetworkAvailable:$true

# Run as current user (no password required for interactive sessions)
$Principal = New-ScheduledTaskPrincipal `
    -UserId $env:USERNAME `
    -LogonType Interactive `
    -RunLevel Highest

# ── Register ──────────────────────────────────────────────────────────────────
try {
    Register-ScheduledTask `
        -TaskName $TaskName `
        -Action $Action `
        -Trigger $Trigger `
        -Settings $Settings `
        -Principal $Principal `
        -Description "Daily Open-Meteo district weather fetch for the Indian Farmer Crop Recommendation System. Skips already-downloaded files." `
        -Force | Out-Null

    Write-Host "✅ Task registered successfully!" -ForegroundColor Green
    Write-Host ""
    Write-Host "  The task will run every day at $RunTime."
    Write-Host "  Progress is logged to: $ProjectRoot\fetch_missing.log"
    Write-Host ""
    Write-Host "  Useful commands:" -ForegroundColor Yellow
    Write-Host "    View task   : Get-ScheduledTask -TaskName '$TaskName'"
    Write-Host "    Run now     : Start-ScheduledTask -TaskName '$TaskName'"
    Write-Host "    Remove task : powershell -File scripts\auto_fetch_scheduler.ps1 -Remove"
    Write-Host ""
} catch {
    Write-Error "Failed to register task: $_"
    Write-Host ""
    Write-Host "  Try running PowerShell as Administrator." -ForegroundColor Yellow
    exit 1
}

# ── Offer to run immediately ──────────────────────────────────────────────────
$RunNow = Read-Host "Run the fetch task right now? (y/N)"
if ($RunNow -match "^[Yy]") {
    Write-Host ""
    Write-Host "Starting task..." -ForegroundColor Cyan
    Start-ScheduledTask -TaskName $TaskName
    Write-Host "✅ Task started. Check fetch_missing.log for progress." -ForegroundColor Green
}

Write-Host ""
Write-Host "============================================================" -ForegroundColor Cyan
Write-Host "  Done. Run 'Get-Content fetch_missing.log -Tail 10' to"
Write-Host "  monitor progress after the task runs."
Write-Host "============================================================" -ForegroundColor Cyan
