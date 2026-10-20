param([string]$Deck, [string]$Destination)
$ErrorActionPreference = 'Stop'
$existing = @(Get-Process POWERPNT -ErrorAction SilentlyContinue)
$application = New-Object -ComObject PowerPoint.Application
$presentation = $null
try {
    $presentation = $application.Presentations.Open($Deck, -1, 0, 0)
    if ($presentation.Slides.Count -ne 32) { throw 'Expected 32 slides' }
    New-Item -ItemType Directory -Path $Destination -Force | Out-Null
    $chartCount = 0
    $tableCount = 0
    foreach ($slide in $presentation.Slides) {
        $file = Join-Path $Destination ('slide-{0:D2}.png' -f $slide.SlideIndex)
        $slide.Export($file, 'PNG', 1920, 1080)
        foreach ($shape in $slide.Shapes) {
            if ($shape.HasChart -eq -1) { $chartCount++ }
            if ($shape.HasTable -eq -1) { $tableCount++ }
        }
        Write-Output ('PowerPoint render {0}/32' -f $slide.SlideIndex)
    }
    Write-Output ('Native charts: {0}, native tables: {1}' -f $chartCount, $tableCount)
}
finally {
    if ($presentation) { $presentation.Close() }
    if ($existing.Count -eq 0) { $application.Quit() }
    [void][Runtime.InteropServices.Marshal]::FinalReleaseComObject($application)
}
