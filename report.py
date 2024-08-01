from twilio.rest import Client


def report_event(event):
    """
    It will send a sms alert notification to the user's phone
    
    """
    #! Before pushing the code, make sure to delete personal info from this file
    
    
    account_sid = '' # Note For this to work, you must create your own twilio account
    auth_token = ''
    client = Client(account_sid, auth_token)

    message = client.messages.create(
    from_='', # twilio account number
    body=f'Security Alert:{event}',
    to='' # your phone number
    )
    with open("current_reports.txt", 'w') as f:
        f.write(f"System: Event-Alert: {event}")
        f.close()
    print(message.sid)


if __name__ == '__main__':
    report_event("Test")