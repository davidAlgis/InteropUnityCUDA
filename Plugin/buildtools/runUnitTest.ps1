
. $PSScriptRoot\dependencies.ps1

function WaitUnity([System.Diagnostics.ProcessStartInfo] $processInfo, $targetName, $unityProjectPath) {
    $appdata = "$env:LOCALAPPDATA"
    $logPath = "$appdata\Unity\Editor\Editor.log"
    $processLogsPath = "$unityProjectPath\TestResults"

    $processInfo.FileName = "$Env:UNITY_2021_3_17\Unity.exe"
    $processInfo.RedirectStandardError = "$processLogsPath\Error.log"
    $processInfo.RedirectStandardOutput = "$processLogsPath\Output.log"
    $processInfo.UseShellExecute = $false

    $process = New-Object System.Diagnostics.Process
    $process.StartInfo = $processInfo
    $process.Start() | Out-Null
    Write-Host "Starting Unity project at $(Resolve-Path $unityProjectPath) (Id = $($process.Id)) ..."

    $time = 0
    $displayTime = 10
    while (-Not ($process.HasExited)) {
        if (-not (Get-Process -Id $process.Id -ErrorAction SilentlyContinue)) {
            $process | Stop-Process -Force
        }
        else {
            Write-Host " Unity (Id = $($process.Id)) work in progress... Don't run away ! (time : $time s)"
            Start-Sleep -Seconds $displayTime;
            $time += $displayTime
        }


        # If a key was pressed during the loop execution, check to see if it was CTRL-C (aka "3")
        if ($Host.UI.RawUI.KeyAvailable -and (3 -eq [int]$Host.UI.RawUI.ReadKey("AllowCtrlC,IncludeKeyUp,NoEcho").Character)) {
            Write-Host ""
            Write-Warning "CTRL-C was used - Shutting down any running jobs before exiting the script."

            # Force end of started process
            $process | Stop-Process -Force
        }
    }

    if ($process.HasExited) {
        # Keep a local copy of Unity logs for each targets
        $unityLogsCopy = "$processLogsPath\$targetName.log"
        Copy-Item $logPath $unityLogsCopy

        # Unity does not send valid keycode at the end of process operation
        # Instead use Unity ouput to try to get information about the operation
        if ($null -ne $(Select-String -Path $logPath -Pattern "another Unity instance is running" -SimpleMatch) -Or $process.ExitCode -ne 0) {
            Write-Host "Unity report an error. More information can be found in Unity editor logs." -ForegroundColor Red

            if ($process.ExitCode -eq 1) {
                Write-Host "Unity builder command line interface report and error. Operation canceled" -ForegroundColor Red
            }
        }
        else {
            $time = $process.ExitTime
            Write-Host "Unity operation done in $time seconds" -ForegroundColor Green
        }

        # Logs for each target
        Write-Host "Logs available at $(Resolve-Path $unityLogsCopy)"
    }
    else {
        Write-Host "User cancel request." -ForegroundColor Yellow
    }

    return $process.ExitCode
}



function RunUnitTestWithAPI($apiName) {
    [OutputType([Boolean])]
    $xmlPath = "$Env:INTEROP_UNITY_CUDA_UNITY_PROJECT_ROOT\TestResults\test_results.xml"

    #if we see a previous xml containing result, we delete it
    if ((Test-Path -Path $xmlPath -PathType Leaf) -eq 1) {
        Remove-Item $xmlPath
    }

    #https://docs.unity3d.com/Packages/com.unity.test-framework@1.1/manual/reference-command-line.html
    $cmdArgs = "-runTests", "-projectPath", "`"$Env:INTEROP_UNITY_CUDA_UNITY_PROJECT_ROOT`"",
    "-batchmode",
    "-force-$apiName",
    "-testResults", "`"$xmlPath`"",
    "-testPlatform", "PlayMode"



    $pinfo = New-Object System.Diagnostics.ProcessStartInfo
    $pinfo.Arguments = $cmdArgs
    WaitUnity $pinfo "UnitTest_InteropUnityCUDA" $Env:INTEROP_UNITY_CUDA_UNITY_PROJECT_ROOT

    [xml]$c = (Get-Content $xmlPath)
    # Display test results
    $details = $c.'test-run'.'test-suite'.'test-suite'.'test-suite'.'test-case' | Select-Object name, duration, result
    $details | Format-Table | Out-String | ForEach-Object { Write-Host $_ }
    Write-Host "Passed " $c.'test-run'.'passed' "/"  $c.'test-run'.'total' 
    $resultTest = $c.'test-run'.'passed' -eq $c.'test-run'.'total'
    if ($resultTest -eq $True) {
        Write-Host "Unit tests with $apiName have passed !" -ForegroundColor Green
    }
    else {
        Write-Host "Unit tests with $apiName have failed !" -ForegroundColor Red
          
    }
    $resultTest
}

$supportedGraphicsAPI = @("glcore", "d3d11") #, "d3d12"

Write-Host ""
Write-Host "------------------------------" -ForegroundColor DarkMagenta
Write-Host "Running : Unit tests" -ForegroundColor DarkMagenta
Write-Host "------------------------------" -ForegroundColor DarkMagenta

$testsPassed = $True
foreach ($api in $supportedGraphicsAPI) {
    Write-Host "Run test with graphics api : $api"
    $result = RunUnitTestWithAPI $api
    $testsPassed = $testsPassed -and $result[1]
}

if ($testsPassed -eq $True) {
    Write-Host "All unit tests have passed !" -ForegroundColor Green
}
else {
    Write-Host "Some Unit tests with $apiName have failed !" -ForegroundColor Red
          
}

Exit $testsPassed