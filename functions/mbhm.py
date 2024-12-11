import numpy as np
from torch.utils.data import Dataset, DataLoader
from dotenv import dotenv_values
import sqlite3
from BearLLM.functions.dcn import dcn


def get_split_id_list(id_list: list, mode: str):
    split_ratio = [0.7, 0.2, 0.1]
    list_length = len(id_list)
    length_list = [int(list_length * ratio) for ratio in split_ratio]
    if mode == 'train':
        return id_list[:length_list[0]]
    elif mode == 'val':
        return id_list[length_list[0]:length_list[0] + length_list[1]]
    elif mode == 'test':
        return id_list[length_list[0] + length_list[1]:]
    else:
        if mode != 'all':
            print('Invalid mode, return all data')
        return id_list


def get_cid_list():
    conn = sqlite3.connect(dotenv_values()['MBHM_INDEX_DB'])
    cursor = conn.cursor()
    cursor.execute('SELECT cid FROM cid_info')
    cid_list = cursor.fetchall()
    conn.close()
    return [cid[0] for cid in cid_list]


def get_fault_free_uuid_list(cid: int, mode='train'):
    conn = sqlite3.connect(dotenv_values()['MBHM_INDEX_DB'])
    cursor = conn.cursor()
    cursor.execute(f'SELECT uuid FROM vibration WHERE cid = {cid} AND label = 0 ORDER BY uuid')
    uuid_list = cursor.fetchall()
    conn.close()
    uuid_list = [uuid[0] for uuid in uuid_list]
    return get_split_id_list(uuid_list, mode)


def load_vibration_index_db(mode='all'):
    conn = sqlite3.connect(dotenv_values()['MBHM_INDEX_DB'])
    cursor = conn.cursor()
    cursor.execute('SELECT uuid, label, cid FROM vibration ORDER BY uuid')
    vibration_list = cursor.fetchall()
    conn.close()
    return get_split_id_list(vibration_list, mode)


def load_text_db():
    conn = sqlite3.connect(dotenv_values()['MBHM_INDEX_DB'])

    text_dict = {}
    for label in range(10):
        text_dict[label] = {}
        for tid in range(4):
            text_dict[label][tid] = []

    cursor = conn.cursor()
    cursor.execute('SELECT * FROM text ORDER BY id')
    text_list = cursor.fetchall()
    conn.close()

    for text in text_list:
        label = text[1]
        tid = text[2]
        text_dict[label][tid].append((text[3], text[4]))

    return text_dict


class FaultFreeDataset:
    def __init__(self, mode='train'):
        self.cid_list = get_cid_list()
        self.uuid_dict = {}
        for cid in self.cid_list:
            self.uuid_dict[cid] = get_fault_free_uuid_list(cid, mode)

    def get_uuid(self, cid: int):
        uuid = self.uuid_dict[cid].pop()
        self.uuid_dict[cid].insert(0, uuid)
        return uuid


class MBHMVibrationDataset(Dataset):
    def __init__(self, mode='all'):
        self.vibration_list = load_vibration_index_db(mode)
        self.length = len(self.vibration_list)
        self.ref_dataset = FaultFreeDataset('train')

    def __len__(self):
        return self.length

    def __getitem__(self, idx):
        uuid, label, cid = self.vibration_list[idx]
        ref_uuid = self.ref_dataset.get_uuid(cid)
        query_data = np.load(f'{dotenv_values()['MBHM_DATA_DIR']}/{uuid}.npy')
        query_data = dcn(query_data)
        ref_data = np.load(f'{dotenv_values()['MBHM_DATA_DIR']}/{ref_uuid}.npy')
        ref_data = dcn(ref_data)
        res_data = query_data - ref_data
        rv = np.array([query_data, ref_data, res_data])
        return rv, label


def mbhm_vibration_dataloader(mode='all', batch_size=1024, shuffle=True, num_workers=0):
    dataset = MBHMVibrationDataset(mode)
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers)


class MBHMDataset(Dataset):
    def __init__(self, mode='all'):
        self.vibration_list = load_vibration_index_db(mode)
        self.length = len(self.vibration_list) * 4
        self.text_dict = load_text_db()

    def __len__(self):
        return self.length

    def __getitem__(self, idx):
        uuid, label, cid = self.vibration_list[idx // 4]
        xv = np.load(f'{dotenv_values()['MBHM_DATA_DIR']}/{uuid}.npy')
        t_id = idx % 4
        text = self.text_dict[label][t_id].pop()
        self.text_dict[label][t_id].insert(0, text)
        xt, gt = text
        return xv, label, cid, xt, gt