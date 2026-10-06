# Schedules bsnn-collect to run every weekday at 19:30 UK time.
#
# 19:30 UK is 14:30 New York for most of the year and 15:30 New York in the weeks
# when the UK and US clocks change on different dates, so it always lands inside
# the US options session. The task runs only while this user is logged on (a
# locked screen is fine), on mains or battery, and runs at the next opportunity if
# the laptop was asleep at 19:30 (bsnn-collect then refuses to save anything if the
# US market has closed by then). It does not wake the laptop.
#
# Run from the repository root:   .\scripts\register-daily-collection.ps1
# Remove it again with:           Unregister-ScheduledTask -TaskName "BSNN daily option snapshots"

$ErrorActionPreference = "Stop"
$repo = Split-Path $PSScriptRoot -Parent
$exe = Join-Path $repo ".venv\Scripts\bsnn-collect.exe"
if (-not (Test-Path $exe)) {
    throw "bsnn-collect not found at $exe. Run: .venv\Scripts\python.exe -m pip install -e `".[dev]`""
}

$action = New-ScheduledTaskAction -Execute $exe -WorkingDirectory $repo
$trigger = New-ScheduledTaskTrigger -Weekly -DaysOfWeek Monday, Tuesday, Wednesday, Thursday, Friday -At "19:30"
# Windows' defaults skip scheduled tasks on battery; a laptop should collect either way (a run takes ~2 minutes).
$settings = New-ScheduledTaskSettingsSet -StartWhenAvailable -AllowStartIfOnBatteries -DontStopIfGoingOnBatteries `
    -ExecutionTimeLimit (New-TimeSpan -Minutes 30)

Register-ScheduledTask -TaskName "BSNN daily option snapshots" -Action $action -Trigger $trigger `
    -Settings $settings -Force `
    -Description "Saves option chains for the watchlist proxies to data\options (log: data\collect.log)."
