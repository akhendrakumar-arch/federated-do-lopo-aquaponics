# Re-run the whole study (several tens of minutes). Run from anywhere:  .\run_all.ps1
$ErrorActionPreference = "Continue"
$root = $PSScriptRoot
function Step($name, [scriptblock]$cmd) {
  Write-Host "=== $name" -ForegroundColor Cyan
  & $cmd
  if ($LASTEXITCODE -ne 0) { throw "step failed: $name (exit $LASTEXITCODE)" }
}
Set-Location "$root\code"
# audit gate: with all candidate ponds it is EXPECTED to fail (documents the duplicate), after excluding Pond 1 it must pass
python -W ignore audit_gate.py | Out-Host
Copy-Item "$root\results\audit.json" "$root\results\audit_all_ponds.json" -Force
Step "audit gate (Pond 1 excluded)" { python -W ignore audit_gate.py --exclude 1 }
Step "classification (LOPO)" { python -W ignore lopo_classification.py }
Step "forecasting (LOPO)" { python -W ignore -u lopo_forecast.py }
Step "conformal recalibration" { python -W ignore -u recal_conformal.py }
Step "example trace + dataset summary" { python -W ignore -u extra_data.py }
Step "baselines" { python -W ignore -u lopo_baselines.py }
Step "federated GRU study" { python -W ignore -u fed_forecast.py }
Step "federated linear (exact)" { python -W ignore -u fed_linear.py }
Step "robustness + alarms" { python -W ignore -u robustness_alarm.py }
Write-Host "done: results\*.json regenerated" -ForegroundColor Green
