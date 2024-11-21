import smtplib
import time
from datetime import datetime
from email.mime.text import MIMEText
from io import BytesIO
from subprocess import PIPE, Popen

import pandas as pd

#           _   _   _
#  ___  ___| |_| |_(_)_ __   __ _ ___
# / __|/ _ \ __| __| | '_ \ / _` / __|
# \__ \  __/ |_| |_| | | | | (_| \__ \
# |___/\___|\__|\__|_|_| |_|\__, |___/
#                           |___/

notify_after = 5
interval_secs = 60 * 5
end_time = '2018-01-20 19:00:00'
mail_address = 'logs_weisspl@gmx.de'

#                               _
#  _ __  _ __ ___   ___ ___  __| |_   _ _ __ ___
# | '_ \| '__/ _ \ / __/ _ \/ _` | | | | '__/ _ \
# | |_) | | | (_) | (_|  __/ (_| | |_| | | |  __/
# | .__/|_|  \___/ \___\___|\__,_|\__,_|_|  \___|
# |_|

t_format = '%Y-%m-%d %H:%M:%S'
end = datetime.strptime(end_time, t_format)
n = 0


def send_message(subject, sender, to, message):
    msg = MIMEText(message, 'html')
    msg['Subject'] = subject
    msg['From'] = sender
    msg['To'] = to

    s = smtplib.SMTP('localhost')
    s.send_message(msg)
    s.quit()


while True:
    if not datetime.now() < end:
        send_message(
            subject='end of logs',
            sender='weisspl@kiwi.ims.uni-stuttgart.de',
            to=mail_address,
            message='Logs have ended'
        )
        exit(0)

    process = Popen(
        args="nvidia-smi --query-compute-apps=pid,process_name,used_memory --format=csv",
        stdout=PIPE,
        shell=True
    )

    # csv_stats = BytesIO(b'utilization.gpu [%], memory.used [MiB]\n98 %, 3114 MiB\n99 %, '
    #                     b'3145 MiB\n99 %, 3492 MiB\n100 %, 4879 MiB\n')
    csv_stats = BytesIO(process.communicate()[0])
    df = pd.read_csv(csv_stats)
    n_processes = df.shape[0]

    if n_processes < 4:
        n = n + 1
        if n >= notify_after:
            n = 0
            send_message(
                message=('<font face="monospace, Courier New, Courier">' + time.strftime(t_format) + '\n\n' + str(
                    df) + '\n\n\nLog will end at: \n' + end_time + '</font>').replace('\n', '<br>'),
                sender='weisspl@kiwi.ims.uni-stuttgart.de',
                subject='Kiwi GPU report',
                to=mail_address)

    time.sleep(interval_secs)
