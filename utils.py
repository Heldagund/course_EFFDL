# For progress bar
import os
import sys
import time
from datetime import datetime
# For sending email
from smtplib import SMTP_SSL
from email import encoders
from email.mime.base import MIMEBase
from email.mime.text import MIMEText
from email.mime.image import MIMEImage
from email.mime.audio import MIMEAudio
from email.mime.multipart import MIMEMultipart
import mimetypes

host = 'smtp.gmail.com'
port = 465     # SSL port

user = 'heldagund@gmail.com'
password = 'vkfktnxwrhzuvdaq'

_, term_width = os.popen('stty size', 'r').read().split()
term_width = int(term_width)

TOTAL_BAR_LENGTH = 65.
last_time = time.time()
begin_time = last_time
def progress_bar(current, total, msg=None):
    global last_time, begin_time
    if current == 0:
        begin_time = time.time()  # Reset for new bar.

    cur_len = int(TOTAL_BAR_LENGTH*current/total)
    rest_len = int(TOTAL_BAR_LENGTH - cur_len) - 1

    sys.stdout.write(' [')
    for i in range(cur_len):
        sys.stdout.write('=')
    sys.stdout.write('>')
    for i in range(rest_len):
        sys.stdout.write('.')
    sys.stdout.write(']')

    cur_time = time.time()
    step_time = cur_time - last_time
    last_time = cur_time
    tot_time = cur_time - begin_time

    L = []
    L.append('  Step: %s' % format_time(step_time))
    L.append(' | Tot: %s' % format_time(tot_time))
    if msg:
        L.append(' | ' + msg)

    msg = ''.join(L)
    sys.stdout.write(msg)
    for i in range(term_width-int(TOTAL_BAR_LENGTH)-len(msg)-3):
        sys.stdout.write(' ')

    # Go back to the center of the bar.
    for i in range(term_width-int(TOTAL_BAR_LENGTH/2)+2):
        sys.stdout.write('\b')
    sys.stdout.write(' %d/%d ' % (current+1, total))

    if current < total-1:
        sys.stdout.write('\r')
    else:
        sys.stdout.write('\n')
    sys.stdout.flush()

def format_time(seconds):
    days = int(seconds / 3600/24)
    seconds = seconds - days*3600*24
    hours = int(seconds / 3600)
    seconds = seconds - hours*3600
    minutes = int(seconds / 60)
    seconds = seconds - minutes*60
    secondsf = int(seconds)
    seconds = seconds - secondsf
    millis = int(seconds*1000)

    f = ''
    i = 1
    if days > 0:
        f += str(days) + 'D'
        i += 1
    if hours > 0 and i <= 2:
        f += str(hours) + 'h'
        i += 1
    if minutes > 0 and i <= 2:
        f += str(minutes) + 'm'
        i += 1
    if secondsf > 0 and i <= 2:
        f += str(secondsf) + 's'
        i += 1
    if millis > 0 and i <= 2:
        f += str(millis) + 'ms'
        i += 1
    if f == '':
        f = '0ms'
    return f

def getDuration(then, now = datetime.now(), interval = "default"):

    # Returns a duration as specified by variable interval
    # Functions, except totalDuration, returns [quotient, remainder]

    duration = now - then # For build-in functions
    duration_in_s = duration.total_seconds() 
    
    # def years():
    #   return divmod(duration_in_s, 31536000) # Seconds in a year=31536000.

    # def days(seconds = None):
    #   return divmod(seconds if seconds != None else duration_in_s, 86400) # Seconds in a day = 86400

    def hours(seconds = None):
      return divmod(seconds if seconds != None else duration_in_s, 3600) # Seconds in an hour = 3600

    def minutes(seconds = None):
      return divmod(seconds if seconds != None else duration_in_s, 60) # Seconds in a minute = 60

    def seconds(seconds = None):
      if seconds != None:
        return divmod(seconds, 1)   
      return duration_in_s

    def durationInHours():
        h = hours()
        m = minutes(h[1])
        s = seconds(m[1])
        return "Elapsed time: {} hours, {} minutes and {} seconds".format(int(h[0]), int(m[0]), int(s[0]))

    # def totalDuration():
    #     y = years()
    #     d = days(y[1]) # Use remainder to calculate next variable
    #     h = hours(d[1])
    #     m = minutes(h[1])
    #     s = seconds(m[1])

    #     return "Time between dates: {} years, {} days, {} hours, {} minutes and {} seconds".format(int(y[0]), int(d[0]), int(h[0]), int(m[0]), int(s[0]))

    return {
        # 'years': int(years()[0]),
        # 'days': int(days()[0]),
        'hours': int(hours()[0]),
        'minutes': int(minutes()[0]),
        'seconds': int(seconds()),
        'default': durationInHours()
    }[interval]

def countParameters(model): 
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

def sendMail(subject, message, sender, recipients, recipientAddrs, directory):
    
    content = MIMEMultipart()
    content['Subject'] = subject
    content['From'] = sender
    content['To'] = recipients
    
    content.attach(MIMEText(message, 'plain', _charset='utf-8'))

    for filename in os.listdir(directory):
        path = os.path.join(directory, filename)
        if not os.path.isfile(path):
            continue
        ctype, encoding = mimetypes.guess_type(path)
        if ctype is None or encoding is not None:
            # No guess could be made, or the file is encoded (compressed), so
            # use a generic bag-of-bits type.
            ctype = 'application/octet-stream'
        maintype, subtype = ctype.split('/', 1)
        if maintype == 'text':
            with open(path) as fp:
                # Note: we should handle calculating the charset
                msg = MIMEText(fp.read(), _subtype=subtype)
        elif maintype == 'image':
            with open(path, 'rb') as fp:
                msg = MIMEImage(fp.read(), _subtype=subtype)
        elif maintype == 'audio':
            with open(path, 'rb') as fp:
                msg = MIMEAudio(fp.read(), _subtype=subtype)
        else:
            with open(path, 'rb') as fp:
                msg = MIMEBase(maintype, subtype)
                msg.set_payload(fp.read())
            # Encode the content using Base64
            encoders.encode_base64(msg)
        #Set the filename parameter
        msg.add_header('Content-Disposition', 'attachment', filename=filename)
        content.attach(msg)

    with SMTP_SSL(host, port) as smtp:
        smtp.login(user, password)
        try:
            smtp.sendmail(user, recipientAddrs.split(','), content.as_string())
            print("OK: send mail succeeded")
        except smtplib.SMTPException:
            print("Error: send mail failed")

def GetLastLines(fileName, lineCnt):
    fileSize =  os.path.getsize(fileName)
    with open(fileName, 'rb') as f:
        offset = -8
        last_lines = []
        while -offset < fileSize:
            f.seek(offset, 2)
            lines = f.readlines()
            if len(lines) >= lineCnt + 1:
                last_lines = lines[-lineCnt:]
                break
            offset *= 2
    return last_lines
# if __name__ =='__main__':
#     message = 'Python 测试邮件...'
#     subject = 'Test multipart message'
#     sender = 'Heldagund'
#     recipient = 'self'
#     attachments = ['document.pdf', '1.png']
    
#     to_addrs = 'heldagund@gmail.com,qwptiger@126.com'
#     sendMail(subject, message, sender, recipient, to_addrs, attachments)   