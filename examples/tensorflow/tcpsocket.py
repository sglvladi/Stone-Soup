import socket
import json


class TcpClient:

    def __init__(self, target, port):

        # create an ipv4 (AF_INET) socket object using the tcp protocol (SOCK_STREAM)
        self.client = socket.socket(socket.AF_INET, socket.SOCK_STREAM)

        # connect the client
        # client.connect((target, port))
        self.client.connect((target, port))

    def send(self, data):

        # send some data (in this case a HTTP GET request)
        self.client.send(str.encode(data))

        # receive the response data (4096 is recommended buffer size)
        response = self.client.recv(4096)

        return response.decode()

    def send_request(self):
        request = {"type": "request"}
        request_str = json.dumps(request)
        response = self.send(request_str)
        return json.loads(response)

    def send_command(self, command):
        command_str = json.dumps(command)
        return self.send(command_str)

#
# if __name__ == "__main__":
#
#     client = TcpClient('127.0.0.1', 1563)
#
#     while True:
#
#         # Test sending request
#         print("\nSending request...")
#         response = client.send_request()
#         print(response)
#
#         # Test sending command
#         command = {"type": "command",
#                    "panSpeed": 10,
#                    "tiltSpeed": 20,
#                    "zoomSpeed": 4}
#         print("\nSending command: {}".format(json.dumps(command)))
#         response = client.send_command(command)
