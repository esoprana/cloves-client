def hook(result):
    print('atd')
    import shutil

    if shutil.which('at') is not None and shutil.which('notify-send') is not None:
        import subprocess
        import dateutil.parser
        import datetime
        job_id = result[1]['job_id']

        date = dateutil.parser.parse(result[1]['scheduled'][1])
        date = date + datetime.timedelta(seconds=30)
        date = date.astimezone().strftime('%Y%m%d%H%M.%S')

        subprocess.run(['at', '-t', date], input=f'notify-send "Cloves job {job_id} completed" -t 900000', text=True)
