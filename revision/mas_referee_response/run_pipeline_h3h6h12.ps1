<#
  run_pipeline_h3h6h12.ps1

  Self-contained, crash-resumable orchestrator for the in-sample GPBSV revision.

  Sequence (strictly sequential, one compute at a time, 14 workers each):
    compute h3  ->  MAS(1,3)  ->  compute h6  ->  MAS(1,3,6)  ->  compute h12  ->  MAS(1,3,6,12)

  Robustness:
   * Runs independently of the launching shell: launch it DETACHED and it survives the shell
     closing. A scheduled task (GPBSV_IS_Pipeline_Resume) can re-run it at logon + hourly
     so it also survives a reboot.
   * Idempotent / resumable: re-running skips any horizon whose insample_gpbsv_h{h}_v5.bin
     exists and any MAS whose snapshot exists, and resumes a half-done compute from its cache.
   * Single-instance lock (pipeline.lock); exits silently once pipeline_DONE.txt exists.
   * On each MAS completion writes MAS_h{h}_FINISHED.txt (durable flag) and a best-effort
     desktop notification, so completions are recorded/announced even if nobody is watching.

  Relaunch / resume command:
    powershell.exe -NoProfile -ExecutionPolicy Bypass -File "<package root>\revision\mas_referee_response\run_pipeline_h3h6h12.ps1"
#>

$ErrorActionPreference = 'Continue'

$repo      = (Resolve-Path (Join-Path $PSScriptRoot '..\..')).Path   # package root
$py        = Join-Path $repo '.venv\Scripts\python.exe'         # package-root Poetry venv
$compute   = "$repo\revision\mas_referee_response\compute_insample_gpbsv_v5.py"
$mas       = "$repo\revision\mas_referee_response\mas_referee_variants.py"
$isdir     = "$repo\revision\mas_referee_response\outputs\insample_gpbsv_v5"
$masout    = "$repo\revision\mas_referee_response\outputs"
$snaproot  = "$masout\mas_snapshots"
$masterLog = "$isdir\pipeline_master.log"
$lock      = "$isdir\pipeline.lock"

function Log([string]$m) {
  $ts = (Get-Date).ToString('yyyy-MM-dd HH:mm:ss')
  try { Add-Content -Path $masterLog -Value "$ts  $m" -ErrorAction Stop } catch {}
}

function Notify([string]$title, [string]$msg) {
  # Best-effort on-machine desktop balloon. Never throws into the pipeline.
  try {
    Add-Type -AssemblyName System.Windows.Forms -ErrorAction Stop
    Add-Type -AssemblyName System.Drawing -ErrorAction Stop
    $ni = New-Object System.Windows.Forms.NotifyIcon
    $ni.Icon = [System.Drawing.SystemIcons]::Information
    $ni.BalloonTipTitle = $title
    $ni.BalloonTipText = $msg
    $ni.Visible = $true
    $ni.ShowBalloonTip(30000)
    Start-Sleep -Seconds 8
    $ni.Dispose()
  } catch {}
}

function Count-Cache([int]$h) {
  $d = "$isdir\_window_cache_h$h"
  if (Test-Path $d) { return @(Get-ChildItem "$d\*.pkl" -ErrorAction SilentlyContinue).Count }
  return 0
}

function Compute-Running([int]$h) {
  $procs = Get-CimInstance Win32_Process -Filter "Name='python.exe'" -ErrorAction SilentlyContinue |
    Where-Object { $_.CommandLine -match 'compute_insample_gpbsv' -and $_.CommandLine -match "--horizons\s+$h\b" }
  return @($procs).Count
}

function Ensure-Compute([int]$h) {
  $bin = "$isdir\insample_gpbsv_h${h}_v5.bin"
  if (Test-Path $bin) { Log "h=$h compute: .bin already present -> skip"; return }
  Log "h=$h compute: starting"
  $stuck = 0
  while (-not (Test-Path $bin)) {
    if ((Compute-Running $h) -gt 0) {
      Log "h=$h compute: an external run is active; waiting for its .bin (cache=$(Count-Cache $h)/416)"
      while (((Compute-Running $h) -gt 0) -and -not (Test-Path $bin)) { Start-Sleep -Seconds 120 }
    } else {
      $before = Count-Cache $h
      Log "h=$h compute: launching (cache=$before/416; resumes cached windows)"
      $out   = "$isdir\run_h${h}_pipeline.out"
      $err   = "$isdir\run_h${h}_pipeline.log"
      $cargs = @($compute,'--mode','sampled-stream','--horizons',"$h",'--rows-per-window','4','--n-jobs','14','--skip-provider-audit','--output-dir',$isdir)
      $p = Start-Process -FilePath $py -ArgumentList $cargs -RedirectStandardOutput $out -RedirectStandardError $err -WindowStyle Hidden -PassThru -Wait
      Log "h=$h compute: run exited code=$($p.ExitCode); cache=$(Count-Cache $h)/416"
      if (Test-Path $bin) { break }
      $after = Count-Cache $h
      if ($after -gt $before) { $stuck = 0 } else { $stuck++ }
      if ($stuck -ge 3) { Log "h=$h compute: FATAL - 3 consecutive attempts with no progress; stopping."; throw "compute h$h stuck" }
      Start-Sleep -Seconds 15
    }
  }
  Log "h=$h compute: COMPLETE (.bin present; cache=$(Count-Cache $h)/416)"
}

function Mas-Done([int]$tagH) {
  return (Test-Path "$snaproot\through_h$tagH\mas_is_oos_gpbsv_v5_formatted_a0.67_mc1000000.csv")
}

function Run-Mas([int[]]$hs, [int]$tagH) {
  if (Mas-Done $tagH) { Log "MAS through h${tagH}: snapshot exists -> skip"; return }
  $hstr = ($hs -join ' ')
  Log "MAS through h${tagH}: running --horizons $hstr --include-insample-gpbsv"
  $margs = @($mas,'--horizons') + ($hs | ForEach-Object { "$_" }) + @('--include-insample-gpbsv')
  $out = "$isdir\mas_through_h${tagH}.out"
  $err = "$isdir\mas_through_h${tagH}.log"
  $p = Start-Process -FilePath $py -ArgumentList $margs -RedirectStandardOutput $out -RedirectStandardError $err -WindowStyle Hidden -PassThru -Wait
  if ($p.ExitCode -ne 0) { Log "MAS through h${tagH}: FAILED exit $($p.ExitCode) (see $err) -- continuing pipeline"; return }
  $snapDir = "$snaproot\through_h$tagH"
  New-Item -ItemType Directory -Force -Path $snapDir | Out-Null
  Copy-Item "$masout\mas_is_oos_gpbsv_v5_*a0.67_mc1000000.*" -Destination $snapDir -Force -ErrorAction SilentlyContinue
  # Durable completion flag (polled by the scheduled task) + on-machine notice.
  Set-Content -Path "$isdir\MAS_h${tagH}_FINISHED.txt" -Value ((Get-Date).ToString('yyyy-MM-dd HH:mm:ss') + "  MAS through h$tagH complete.  Table: $masout\mas_is_oos_gpbsv_v5_a0.67_mc1000000.csv  Snapshot: $snapDir")
  Notify "GPBSV MAS h$tagH done" "MAS through h$tagH finished; table updated (horizons up to $tagH)."
  Log "MAS through h${tagH}: COMPLETE; snapshot -> $snapDir; FINISHED flag + desktop notice written"
}

# ---------------- main ----------------
try {
  if (Test-Path "$isdir\pipeline_DONE.txt") { return }
  if (Test-Path $lock) {
    $lpid = (Get-Content $lock -ErrorAction SilentlyContinue | Select-Object -First 1)
    if ($lpid -and (Get-Process -Id $lpid -ErrorAction SilentlyContinue)) {
      Log "Another orchestrator (PID $lpid) is already running; instance $PID exits."
      return
    }
    Log "Stale lock found (PID $lpid not alive); taking over."
  }
  Set-Content -Path $lock -Value "$PID"
  Log "=== PIPELINE START (orchestrator PID $PID) ==="

  Ensure-Compute 3
  Run-Mas @(1,3) 3
  Ensure-Compute 6
  Run-Mas @(1,3,6) 6
  Ensure-Compute 12
  Run-Mas @(1,3,6,12) 12

  Set-Content -Path "$isdir\pipeline_DONE.txt" -Value ((Get-Date).ToString('yyyy-MM-dd HH:mm:ss') + "  all horizons + MAS complete")
  Log "=== PIPELINE COMPLETE: h3/h6/h12 computes + MAS done; final table = outputs\mas_is_oos_gpbsv_v5_a0.67_mc1000000.* (snapshots in mas_snapshots\through_h12) ==="
}
catch {
  Log "PIPELINE ERROR: $($_.Exception.Message)"
}
finally {
  if (Test-Path $lock) {
    $lpid = (Get-Content $lock -ErrorAction SilentlyContinue | Select-Object -First 1)
    if ("$lpid" -eq "$PID") { Remove-Item $lock -Force -ErrorAction SilentlyContinue }
  }
}
