$Id$

cd C:\Users\lambert.wixson\datasets\kaggle_carvana\project_front_vs_back


copy ..\train\small-color\*_01.png .\train-color\yaw01
copy ..\train\small-color\*_05.png .\train-color\yaw05
copy ..\train\small-color\*_09.png .\train-color\yaw09
copy ..\train\small-color\*_13.png .\train-color\yaw13

#copy ..\extratrain\small-color\*_01.png .\train-color\yaw01
#copy ..\extratrain\small-color\*_09.png .\train-color\yaw09

copy ..\validate\small-color\*_01.png .\validate-color\yaw01
copy ..\validate\small-color\*_05.png .\validate-color\yaw05
copy ..\validate\small-color\*_09.png .\validate-color\yaw09
copy ..\validate\small-color\*_13.png .\validate-color\yaw13

copy ..\test\small-color\*_01.png .\test-color\yaw01
copy ..\test\small-color\*_05.png .\test-color\yaw05
copy ..\test\small-color\*_09.png .\test-color\yaw09
copy ..\test\small-color\*_13.png .\test-color\yaw13

cd C:\Users\lambert.wixson\datasets\kaggle_carvana\test\full
$a = ls *.jpg 
$count = 0
$a | sort -Property Name | foreach { $count += 1 
$last = $_.Name
move $_ ..\..\validate\full
if ($count -ge 11888) {
write-host "Last file moved was $last"
break
}
}

$a = ls *.jpg 
$count = 0
$a | sort -Property Name | foreach { $count += 1 
$last = $_.Name
move $_ ..\..\validate\small-color
if ($count -ge 1486) {
write-host "Last file moved was $last"
break
}
}