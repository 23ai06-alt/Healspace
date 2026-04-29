db = mysql.connector.connect(
    host="localhost",
    user="root",
    password="simu@26",  
    database="healspace",
    auth_plugin='mysql_native_password'   # 👈 ab sahi hai
)

