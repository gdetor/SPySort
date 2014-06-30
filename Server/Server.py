import socket
import select
import pickle
import numpy as np
import matplotlib.pylab as plt


def reset_():
    start = 0
    end = 100
    count = 0
    return start, end, count


def plot_(data):
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.plot(data, color='.75')
    plt.show()


if __name__ == '__main__':
    s = socket.socket()          # Create a socket object
    host = socket.gethostname()  # Get local machine name
    port = 4242                  # Reserve a port
    s.bind((host, port))  # Bind the socket to the host/port

    print "Listening on port {p}...".format(p=port)
    s.listen(5)                 # Now wait for client connection.

    p = 0
    start, end, count_p = reset_()
    while True:
        try:
            client, addr = s.accept()
            ready = select.select([client, ], [], [], 2)
            if ready[0]:
                data = client.recv(4096)
                data = pickle.loads(data)
                if type(data) is dict:
                    p = data['sentinel_']
                    cumul_data = np.zeros((p*100,))
                    print 'Size of data array has been set to {}'.format(p)
                if isinstance(data, (np.ndarray, np.generic)):
                    cumul_data[start:end] = data.copy()
                    start += 100
                    end += 100
                    count_p += 1
                if cumul_data.shape[0] == p * 100 and count_p == p:
                    plot_(cumul_data)
                    start, end, count_p = reset_()
        except KeyboardInterrupt:
            print
            print "You pressed Ctrl+C!"
            print "Quitting ..."
            break
