# Creates the cron-job.org jobs that start the bsnn-data workflows on time.
#
# GitHub's own schedules can start hours late, so cron-job.org (free) calls GitHub's
# "run workflow" API at the exact New York times instead. Each call runs the workflow
# with post_now=false, so the slot checks in bsnn-levels still post each slot once.
#
# Needs two keys, both typed into hidden prompts (never stored in a file):
#   1. A GitHub fine-grained token: only the bsnn-data repository, Actions: Read and write.
#   2. A cron-job.org API key: cron-job.org > Settings > API.
#
# Run from the repository root:   .\scripts\register-cron-jobs.ps1
# Running it again updates the times of jobs that already exist (matched by title) when they differ,
# and deletes retired ones. The GitHub token is only asked for when a job has to be created.

$ErrorActionPreference = "Stop"
$repo = "issyk0704/bsnn-data"
$weekdays = @(1, 2, 3, 4, 5)  # cron-job.org counts 0 = Sunday

$jobs = @(
    @{ Title = "BSNN daily levels"; Workflow = "levels.yml";   Hours = @(9);      Minutes = @(15); Inputs = @{ post_now = $false } }
    # A new card at 10:45 and 14:45, just before the 10:50 and 14:50 ICT macros (no 09:45 update since 17 Nov 2026).
    @{ Title = "BSNN levels updates"; Workflow = "levels.yml"; Hours = @(10, 12, 14); Minutes = @(45); Inputs = @{ post_now = $false } }
    @{ Title = "BSNN collect";      Workflow = "collect.yml";  Hours = @(16);     Minutes = @(30); Inputs = @{ force = $false } }
)

$githubToken = $null
$cronKey = Read-Host "cron-job.org API key" -AsSecureString | ConvertFrom-SecureString -AsPlainText
$api = @{ Authorization = "Bearer $cronKey" }

# Jobs from earlier schedules, deleted if they still exist.
$retired = @("BSNN 0DTE :00", "BSNN 0DTE :30")  # 0DTE posts, replaced by day/week/month levels in the 09:15 post

$current = (Invoke-RestMethod "https://api.cron-job.org/jobs" -Headers $api).jobs
foreach ($old in $current | Where-Object { $retired -contains $_.title }) {
    Invoke-RestMethod "https://api.cron-job.org/jobs/$($old.jobId)" -Method Delete -Headers $api | Out-Null
    Write-Host "Deleted retired job: $($old.title)"
    Start-Sleep -Seconds 1
}

foreach ($job in $jobs) {
    $found = $current | Where-Object { $_.title -eq $job.Title } | Select-Object -First 1
    if ($found) {
        $hours = @($found.schedule.hours) -join ","
        $minutes = @($found.schedule.minutes) -join ","
        if ($hours -eq ($job.Hours -join ",") -and $minutes -eq ($job.Minutes -join ",")) {
            Write-Host "Exists already, times match: $($job.Title)"
            continue
        }
        $patch = @{ job = @{ schedule = @{
            timezone  = "America/New_York"
            expiresAt = 0
            hours     = $job.Hours
            minutes   = $job.Minutes
            mdays     = @(-1)
            months    = @(-1)
            wdays     = $weekdays
        } } } | ConvertTo-Json -Depth 4
        Invoke-RestMethod "https://api.cron-job.org/jobs/$($found.jobId)" -Method Patch -Headers $api `
            -ContentType "application/json" -Body $patch | Out-Null
        Write-Host "Updated times: $($job.Title) (job $($found.jobId)) hours $hours -> $($job.Hours -join ','), minutes $minutes -> $($job.Minutes -join ',')"
        Start-Sleep -Seconds 1
        continue
    }
    if (-not $githubToken) {
        $githubToken = Read-Host "GitHub token (bsnn-data, Actions read/write)" -AsSecureString | ConvertFrom-SecureString -AsPlainText
    }
    $body = @{
        job = @{
            title         = $job.Title
            url           = "https://api.github.com/repos/$repo/actions/workflows/$($job.Workflow)/dispatches"
            enabled       = $true
            saveResponses = $true
            requestMethod = 1  # POST
            schedule      = @{
                timezone  = "America/New_York"
                expiresAt = 0
                hours     = $job.Hours
                minutes   = $job.Minutes
                mdays     = @(-1)
                months    = @(-1)
                wdays     = $weekdays
            }
            extendedData  = @{
                headers = @{
                    "Accept"               = "application/vnd.github+json"
                    "Authorization"        = "Bearer $githubToken"
                    "X-GitHub-Api-Version" = "2022-11-28"
                    "Content-Type"         = "application/json"
                }
                body    = (@{ ref = "main"; inputs = $job.Inputs } | ConvertTo-Json -Compress)
            }
            notification  = @{ onFailure = $true; onSuccess = $false; onDisable = $true }
        }
    } | ConvertTo-Json -Depth 6
    $created = Invoke-RestMethod "https://api.cron-job.org/jobs" -Method Put -Headers $api `
        -ContentType "application/json" -Body $body
    Write-Host "Created: $($job.Title) (job $($created.jobId))"
    Start-Sleep -Seconds 1  # the cron-job.org API allows about one request a second
}
