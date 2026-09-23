param(
    [Parameter(Mandatory = $true, Position = 0)]
    [string]$Root,
    [string]$Config = "npf.count.in",
    [switch]$DryRun
)

$ErrorActionPreference = "Stop"
$rootPath = (Resolve-Path -LiteralPath $Root).Path
$structures = @(Get-ChildItem -LiteralPath $rootPath -Filter POSCAR -File -Recurse | Sort-Object FullName)
Write-Host ("Found {0} POSCAR files under {1}" -f $structures.Count, $rootPath)
if ($structures.Count -eq 0) { exit 2 }

foreach ($item in $structures) {
    Write-Host ("Counting {0}" -f $item.FullName)
    if ($DryRun) { continue }
    $log = Join-Path $item.DirectoryName "site_count_batch.log"
    & python -m npf.count $item.FullName --config $Config *> $log
    if ($LASTEXITCODE -ne 0) {
        Write-Warning ("Failed: {0}; see {1}" -f $item.FullName, $log)
    }
}
