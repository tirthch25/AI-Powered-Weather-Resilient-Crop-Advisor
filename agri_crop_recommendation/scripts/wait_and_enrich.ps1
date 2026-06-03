# wait_and_enrich.ps1
# Waits until Gemini free-tier quota resets (midnight UTC = 5:30 AM IST)
# then automatically starts the enrichment pipeline.

$ScriptDir = Split-Path -Parent $MyInvocation.MyCommand.Path
$PackageRoot = Split-Path -Parent $ScriptDir
Set-Location $PackageRoot

# Midnight UTC in local time
$now = Get-Date
$midnightUtc = $now.Date.AddHours(5).AddMinutes(31)   # 5:30 AM IST = 00:00 UTC
if ($now -ge $midnightUtc) {
    $midnightUtc = $midnightUtc.AddDays(1)
}
$waitSecs = [int]($midnightUtc - $now).TotalSeconds

Write-Host "=============================================="
Write-Host "  Current time : $($now.ToString('yyyy-MM-dd HH:mm:ss'))"
Write-Host "  Quota resets : $($midnightUtc.ToString('yyyy-MM-dd HH:mm:ss')) IST"
Write-Host "  Waiting      : $([math]::Round($waitSecs/3600, 1)) hours ($waitSecs seconds)"
Write-Host "=============================================="

Start-Sleep -Seconds $waitSecs

Write-Host ""
Write-Host "Quota reset window reached. Starting enrichment..."
Write-Host "=============================================="

python scripts/enrich_regional_crops.py --only-missing --delay 5
