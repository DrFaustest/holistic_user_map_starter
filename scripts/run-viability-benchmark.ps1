param(
    [Parameter(ValueFromRemainingArguments = $true)]
    [string[]]$CliArgs
)

function Test-Python311 {
    param(
        [string]$Command,
        [string[]]$PrefixArgs = @()
    )

    try {
        $version = & $Command @PrefixArgs -c "import sys; print(f'{sys.version_info.major}.{sys.version_info.minor}')"
        return $LASTEXITCODE -eq 0 -and $version.Trim() -eq "3.11"
    }
    catch {
        return $false
    }
}

$candidates = @(
    @{ Command = "py"; PrefixArgs = @("-3.11") },
    @{ Command = "python3.11"; PrefixArgs = @() },
    @{ Command = "C:/Users/scott/AppData/Local/Programs/Python/Python311/python.exe"; PrefixArgs = @() },
    @{ Command = "python"; PrefixArgs = @() }
)

foreach ($candidate in $candidates) {
    if (Test-Python311 -Command $candidate.Command -PrefixArgs $candidate.PrefixArgs) {
        & $candidate.Command @($candidate.PrefixArgs + @("-m", "app.cli", "viability-benchmark") + $CliArgs)
        exit $LASTEXITCODE
    }
}

Write-Error "Python 3.11 was not found. Use 'py -3.11 -m app.cli viability-benchmark' or install Python 3.11 before running the benchmark."
exit 1