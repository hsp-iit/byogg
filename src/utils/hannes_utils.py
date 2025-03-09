from libs.pyHannesAPI.pyHannesAPI import pyHannes_commands

def hannes_init(hannes, max_attempts=3, control_modality='CONTROL_EMG_CAMERA'):
    assert control_modality in [*pyHannes_commands.HControlModality.__dict__, 'CONTROL_FIRMWARE']
    while True:
        try:
            hannes.connect()
        except Exception as e:
            max_attempts -= 1
            if max_attempts == 0:
                raise e
            else:
                print(
                    "Connection to Hannes failed, remaining "
                    f"attempts: {max_attempts}"
                )
        else:
            break
    
    if control_modality != 'CONTROL_FIRMWARE':
        hannes.set_control_modality(
            pyHannes_commands.HControlModality.__dict__[control_modality]
        )
        hannes.enable_reply_measurements()
        hannes.enable_reply_quaternions()
        hannes.enable_reply_gravity()
        hannes.enable_reply_emg()
        hannes.enable_reply_end_data_stream()
    else:
        hannes.enable_reply_measurements()
        hannes.enable_reply_emg()
        hannes.enable_reply_end_data_stream()