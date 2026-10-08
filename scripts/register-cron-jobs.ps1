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
# Running it again skips jobs that already exist (matched by title).

$ErrorActionPreference = "Stop"
$repo = "issyk0704/bsnn-data"
$weekdays = @(1, 2, 3, 4, 5)  # cron-job.org counts 0 = Sunday

$jobs = @(
    @{ Title = "BSNN daily levels"; Workflow = "levels.yml";   Hours = @(9);      Minutes = @(15); Inputs = @{ post_now = $false } }
    @{ Title = "BSNN 0DTE :00";     Workflow = "zero-dte.yml"; Hours = @(10, 15); Minutes = @(0);  Inputs = @{ post_now = $false } }
    @{ Title = "BSNN 0DTE :30";     Workflow = "zero-dte.yml"; Hours = @(11, 13); Minutes = @(30); Inputs = @{ post_now = $false } }
    @{ Title = "BSNN collect";      Workflow = "collect.yml";  Hours = @(14);     Minutes = @(30); Inputs = @{ force = $false } }
)

$githubToken = Read-Host "GitHub token (bsnn-data, Actions read/write)" -AsSecureString | ConvertFrom-SecureString -AsPlainText
$cronKey = Read-Host "cron-job.org API key" -AsSecureString | ConvertFrom-SecureString -AsPlainText
$api = @{ Authorization = "Bearer $cronKey" }

$existing = (Invoke-RestMethod "https://api.cron-job.org/jobs" -Headers $api).jobs.title
foreach ($job in $jobs) {
    if ($existing -contains $job.Title) {
        Write-Host "Exists already, skipped: $($job.Title)"
        continue
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
