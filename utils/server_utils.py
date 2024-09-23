import os, sys, shutil
import logging

import paramiko
from paramiko import SSHClient
from stat import S_ISDIR, S_ISREG
from tqdm import tqdm



class ServerManager:
    """
    This class allows to download and upload files, folders to/from server.
    """
    def __init__(self,
                 ssh_key_path,
                 hostname="85.254.226.77",
                 port=22,
                 username="misik01",
                 ):
        self._ssh_key_path = ssh_key_path
        self._hostname = hostname
        self._port = port
        self._username = username

        # Set logging
        logging.basicConfig(format="%(filename)s:%(levelname)s - %(message)s", level=logging.ERROR)
        self.logger = logging.getLogger(__name__)
        self.logger.setLevel(logging.INFO)

        self._ssh = SSHClient()
        self._ssh.set_missing_host_key_policy(paramiko.AutoAddPolicy())
        self._ssh.load_system_host_keys()
        try:
            self._ssh.connect(
                hostname=self._hostname,
                port=self._port,
                username=self._username,
                key_filename=ssh_key_path
            )
            self.logger.info(f"Connection to server {self._hostname}: SUCCESS")
        except Exception as e:
            print(f"Exception in  ssh.connect():")
            print(e)
            sys.exit()

        # SCPCLient takes a paramiko transport as its only argument
        self.cbk, self.pbar = cbk, pbar = self.tqdm_wrap_viewbar(ascii=True, unit='b', unit_scale=True)

    def __exit__(self):
        self._ssh.close()

    @staticmethod
    def tqdm_wrap_viewbar(*args, **kwargs):
        pbar = tqdm(*args, **kwargs)  # make a progressbar
        last = [0]  # last known iteration, start at 0

        def view_bar(a, b):
            pbar.total = int(b)
            pbar.update(int(a - last[0]))  # update pbar with increment
            last[0] = a  # update last known iteration

        return view_bar, pbar  # return callback, tqdmInstance

    def execute_command(self, cmd):
        _stdin, _stdout, _stderr = self._ssh.exec_command(cmd)
        return _stdout.read().decode()

    def list_folders(self, path=None):
        folders = []
        ftp = self._ssh.open_sftp()
        for entry in ftp.listdir_attr(path=path):  # Return all files in path
            mode = entry.st_mode
            if S_ISDIR(mode):
                folders.append(entry.filename)
        return folders

    def list_files(self, path=None):
        files = []
        ftp = self._ssh.open_sftp()
        for entry in ftp.listdir_attr(path=path):  # Return all files in path
            mode = entry.st_mode
            if S_ISREG(mode):
                files.append(entry.filename)
        return files

    def read_from_server(self, source, destination, overwrite=False):
        """
        This function reads files/folders from server and copy to local disk
        :param source: list of files or file
        :param destination: destination folder
        :param overwrite: if True, it will delete destination data before copying
        :return:
        """
        if overwrite and os.path.exists(destination):
            #  Delete that folder
            shutil.rmtree(destination)

        if not os.path.exists(destination):
            os.mkdir(destination)

        if not isinstance(source, list):
            source = list(source.split())

        with self._ssh.open_sftp() as sftp:
            for s in source:
                if os.path.exists(os.path.join(destination, os.path.basename(s))):
                    self.logger.info(f"File {os.path.join(destination, os.path.basename(s))} already exists!")
                    continue
                self.logger.info(f"Reading file {s} from the server to {destination}.")
                sftp.get(remotepath=s, localpath=os.path.join(destination, os.path.basename(s)), callback=self.cbk)


if __name__ == "__main__":
    sm = ServerManager(ssh_key_path="/home/hercogs/.ssh/id_rsa_misik01.pub")
    #print(sm.execute_command("ls /home_beegfs/groups/misik/forestai/drones/missions/44980020033/1/1/06-10-2023/mavic3"))
    fil = [
        "/home_beegfs/groups/misik/forestai/drones/missions/44980020033/1/1/06-10-2023/mavic3/DJI_20231006104656_0867_D.JPG",
        "/home_beegfs/groups/misik/forestai/drones/missions/44980020033/1/1/06-10-2023/mavic3/DJI_20231006104432_0826_D.JPG"
    ]
    sm.read_from_server(fil, "./tmp")