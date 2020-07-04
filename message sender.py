from twilio.rest import Client


# Your Account Sid and Auth Token from twilio.com/console
# DANGER! This is insecure. See http://twil.io/secure
account_sid = 'ACb307221cfb299d2aae4c1aaf03c348bb'
auth_token = '07494a4be8eb25f38fdb9e5095d6be26'
client = Client(account_sid, auth_token)

message = client.messages\
                .create(
                     body="Dangerous",
                     from_='+12673101724',
                     to='+919137485595'
                 )

print(message.sid)