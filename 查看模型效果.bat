@echo off
chcp 65001 >nul
cd /d C:\srtp
echo ========================================
echo          MQ模型测试
echo ========================================
python predict_mq_gas.py --xlsx Gas_Sensors_Measurements.xlsx --index 100
echo.
echo ========================================
echo          氨气模型训练/效果
echo ========================================
python train_ammonia_model.py --data-dir C:\srtp\uci_gas\unzipped
echo.
echo ========================================
echo          丙酮模型训练/效果
echo ========================================
python train_acetone_model.py --data-dir C:\srtp\uci_gas\unzipped
echo.
pause
