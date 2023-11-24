from src.encoder import *
from src.fusion import *
from sklearn.metrics import f1_score
from sklearn.metrics import accuracy_score



class Client():
    def __init__(self,configs,data):
        self.configs = configs
        self.data = data
        self.batch_size = int(configs['fusion']['batch_size'])
        self.dataset = configs['config']['dataset']
        self.raw_train_x = data['x_train']
        self.raw_train_y = data['y_train']
        self.raw_test_x = data['x_test']
        self.raw_test_y = data['y_test']
        self.available_modality = data['info']
        self.id = data['id']
        self.stat = data['statistics']
        self.min = min(self.stat)
        self.max = max(self.stat)
        self.range = self.max - self.min
        self.D = int(configs['config']['D'])

        self.num_train = len(self.raw_train_x)
        self.num_test = len(self.raw_test_x)

        self.train_batchs = []
        self.test_batchs = []

        if self.dataset == "HAR":
            self.all_labels = [0,1,2,3,4,5]
            self.num_readings=9
            self.num_class = 6
            acce_idx = [0,1,2,3,4,5]
            gyro_idx = [6,7,8]
            self.keep_idx = []
            if 'acce' in self.available_modality:
                self.keep_idx += acce_idx
            if 'gyro' in self.available_modality:
                self.keep_idx += gyro_idx
        if self.dataset == "MHEALTH":
            self.all_labels = [0,1,2,3,4,5,6,7,8,9,10,11,12]
            self.num_readings=21
            self.num_class = 13
            acce_idx = [0,1,2,3,4,5,5,7,8]
            gyro_idx = [9,10,11,12,13,14]
            mage_idx = [15,16,17,18,19,20]
            self.keep_idx = []
            if 'acce' in self.available_modality:
                self.keep_idx += acce_idx
            if 'gyro' in self.available_modality:
                self.keep_idx += gyro_idx
            if 'mage' in self.available_modality:
                self.keep_idx += mage_idx

        if self.dataset == "OPP":
            self.all_labels = [0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16]
            self.num_readings=39
            self.num_class = 17
            acce_idx = [i for i in range(24)]
            gyro_idx = [i for i in range(24,39)]
            self.keep_idx = []
            if 'acce' in self.available_modality:
                self.keep_idx += acce_idx
            if 'gyro' in self.available_modality:
                self.keep_idx += gyro_idx

        self.display_info(str(self.available_modality))


    def display_info(self,msg):
        print("Client " + str(self.id) + ": " + msg)


    def init(self):
        self.encoder = Encoder(self.configs,self.min,self.max,self.range)
        self.model = attention_module(self.configs,self.keep_idx)
        self.display_info("Initialized")


    def encode_train(self):
        self.train_encodings = []
        self.train_sample_num = len(self.raw_train_x)
        for i in range(self.train_sample_num):
            enc = self.encoder.encode_one_sample(self.raw_train_x[i])
            if self.dataset == "HAR":
                if "acce" not in self.available_modality:
                    enc[0:6,:] = 0
                if "gyro" not in self.available_modality:
                    enc[6:9,:] = 0
            if self.dataset == "MHEALTH":
                if "acce" not in self.available_modality:
                    enc[0:9,:] = 0
                if "gyro" not in self.available_modality:
                    enc[9:15,:] = 0
                if "mage" not in self.available_modality:
                    enc[15:21,:] = 0
            if self.dataset == "OPP":
                if "acce" not in self.available_modality:
                    enc[0:24,:] = 0
                if "gyro" not in self.available_modality:
                    enc[24:39,:] = 0
            enc = torch.from_numpy(enc)
            self.train_encodings.append(enc)

        assert len(self.train_encodings) == len(self.raw_train_y)
        self.display_info("Train encoded")

    def encode_test(self):
        self.test_encodings = []
        self.test_sample_num = len(self.raw_test_x)
        for i in range(self.test_sample_num):
            enc = self.encoder.encode_one_sample(self.raw_test_x[i])
            if self.dataset == "HAR":
                if "acce" not in self.available_modality:
                    enc[0:6,:] = 0
                if "gyro" not in self.available_modality:
                    enc[6:9,:] = 0
            if self.dataset == "MHEALTH":
                if "acce" not in self.available_modality:
                    enc[0:9,:] = 0
                if "gyro" not in self.available_modality:
                    enc[9:15,:] = 0
                if "mage" not in self.available_modality:
                    enc[15:21,:] = 0
            if self.dataset == "OPP":
                if "acce" not in self.available_modality:
                    enc[0:24,:] = 0
                if "gyro" not in self.available_modality:
                    enc[24:39,:] = 0
            enc = torch.from_numpy(enc)
            self.test_encodings.append(enc)
            
        assert len(self.test_encodings) == len(self.raw_test_y)
        self.display_info("Test encoded")


    def batch_input(self):
        start = 0
        while start < self.num_train:
            batch_data = torch.stack(self.train_encodings[start:start+self.batch_size],dim=0).float()
            batch_label = torch.from_numpy(self.raw_train_y[start:start+self.batch_size])
            self.train_batchs.append((batch_data,batch_label))
            start+=self.batch_size

        start = 0
        while start < self.num_test:
            batch_data = torch.stack(self.test_encodings[start:start+self.batch_size],dim=0).float()
            batch_label = torch.from_numpy(self.raw_test_y[start:start+self.batch_size])
            self.test_batchs.append((batch_data,batch_label))
            start+=self.batch_size

        self.display_info("Inputs batched")




    def eval(self):
        preds = []
        labels = []
        with torch.no_grad():
            for batch in self.train_batchs:
                data = batch[0]
                label = batch[1]
                preds += torch.argmax(self.model(data),dim=1).tolist()
                labels += label.tolist()
        f1 = f1_score(labels, preds, average='weighted',labels=self.all_labels)
        acc = accuracy_score(labels, preds)
        self.display_info("eval F1: {0} - eval Acc: {1}".format(f1,acc))
        return f1, acc


    def test(self):
        preds = []
        labels = []
        with torch.no_grad():
            for batch in self.test_batchs:
                data = batch[0]
                label = batch[1]
                preds += torch.argmax(self.model(data),dim=1).tolist()
                labels += label.tolist()
        f1 = f1_score(labels, preds, average='weighted',labels=self.all_labels)
        acc = accuracy_score(labels, preds)
        self.display_info("test F1: {0} - test Acc: {1}".format(f1,acc))
        return f1, acc


    def train(self,epoch):
        criterion = nn.CrossEntropyLoss()
        optimizer = torch.optim.Adam(self.model.parameters(),lr=0.0003,eps=1e-08,weight_decay=0)
        for epoch in range(epoch):
            running_loss = 0.0
            for batch in self.train_batchs:
                data, label = batch
                optimizer.zero_grad()
                outputs = self.model(data)
                loss = criterion(outputs,label)
                loss.backward()
                optimizer.step()
                running_loss += loss.item()
            self.display_info("Epoch Loss " + str(epoch) + " : " + str(running_loss))



















