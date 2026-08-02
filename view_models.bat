@echo off
cd /d C:\srtp
echo MQ MODEL TEST
python predict_mq_gas.py --xlsx Gas_Sensors_Measurements.xlsx --index 100
echo.
echo AMMONIA MODEL RESULT
python train_ammonia_model.py --data-dir C:\srtp\uci_gas\unzipped
echo.
echo ACETONE MODEL RESULT
python train_acetone_model.py --data-dir C:\srtp\uci_gas\unzipped
echo.
pause
