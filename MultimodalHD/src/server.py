import pickle
from src.client import *
from sklearn.metrics import f1_score
from sklearn.metrics import accuracy_score
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import seaborn as sns
import copy
from datetime import datetime
import os
import csv
import gc






class Server():
    def __init__(self,configs):
        self.configs = configs
        self.exp_name = configs['config']['exp_name']
        self.start_time = datetime.now()
        self.result_path = "results/"+self.exp_name+'_'+datetime.now().strftime("%m_%d_%y_%H_%M_%S")
        os.mkdir(self.result_path)
        self.round = int(configs['config']['round'])
        self.epoch = int(configs['config']['epoch'])
        self.num_client = int(configs['config']['num_client'])
        self.dataset = configs['config']['dataset']
        if self.dataset == 'HAR':
            self.datapath = '../data/har/har_client/'
        if self.dataset == 'MHEALTH':
            self.datapath = '../data/mhealth/mhealth_client/'
        if self.dataset == 'OPP':
            self.datapath = '../data/OPP/OPP_client/'

        if self.dataset == "HAR":
            self.num_readings=9
            self.num_class = 6
        if self.dataset == "MHEALTH":
            self.num_readings=21
            self.num_class = 13
        if self.dataset == "OPP":
            self.num_readings=39
            self.num_class = 17
        self.D = int(configs['config']['D'])

        if configs['config']['aggregation'] == 'nofed':
            self.aggregation = self.aggregate_network_nofed
        elif configs['config']['aggregation'] == 'fedavg':
            self.aggregation = self.aggregate_network_fedavg
        elif configs['config']['aggregation'] == 'fedper':
            self.aggregation = self.aggregate_network_fedper
        elif configs['config']['aggregation'] == 'qkv':
            self.aggregation = self.aggregate_network_qkv
        elif configs['config']['aggregation'] == 'proximity':
            self.aggregation = self.aggregate_network_proximity
            self.temperature = float(configs['config']['temperature'])
        elif configs['config']['aggregation'] == 'attn_proximity':
            self.aggregation = self.aggregate_network_attn_proximity
            self.temperature = float(configs['config']['temperature'])
        else:
            self.display_info("Invalid aggregation method")
            exit()

        self.eval_results_f1=[]
        self.eval_results_acc=[]
        self.test_results_f1=[]
        self.test_results_acc=[]
        self.round_time=[]
        
        if self.configs['config']['epoch_time'] == 'True':
            self.encoding_time = None
            self.epoch_time = []
            self.measure_epoch_time()

    def display_info(self,msg):
        print("Server " + ": " + msg)

    def init_clients(self):
        self.clients = []
        for c in range(self.num_client):
            with open(self.datapath+'client_{0}_data'.format(str(c)), 'rb') as data_file:
                data = pickle.load(data_file)
            self.clients.append(Client(self.configs,data))
            self.clients[c].init()
        data = None

        level_hvs = copy.deepcopy(self.clients[0].encoder.level_hvs)
        id_hvs = copy.deepcopy(self.clients[0].encoder.id_hvs)

        for c in range(self.num_client):
            self.clients[c].encoder.level_hvs = copy.deepcopy(level_hvs)
            self.clients[c].encoder.id_hvs = copy.deepcopy(id_hvs)

        self.num_learnable_parameters = 0
        self.num_static_parameters = 0

        initial_parameters = {}
        self.display_info("========Parameters========")
        for name, param in self.clients[0].model.named_parameters():
            if param.requires_grad:
                self.display_info(name)
                initial_parameters[name] = copy.deepcopy(param.data)
                self.num_learnable_parameters += torch.prod(torch.tensor(param.data.shape))
            if param.requires_grad==False:
                self.display_info("========no grad========")
                self.display_info(name)
                initial_parameters[name] = copy.deepcopy(param.data)
                self.num_static_parameters += torch.prod(torch.tensor(param.data.shape))
                self.display_info("========no grad========")
        for c in self.clients:
            for name, param in c.model.named_parameters():
                param.data = copy.deepcopy(initial_parameters[name])
                pass
        self.display_info("==========================")
        self.display_info('Number of trainable parameters: {0}'.format(self.num_learnable_parameters))
        self.display_info('Number of static parameters: {0}'.format(self.num_static_parameters))
        self.display_info('All client initialized')


    def initial_state_check(self):
        level_hvs = self.clients[0].encoder.level_hvs
        id_hvs = self.clients[0].encoder.id_hvs
        min = self.clients[0].min
        max = self.clients[0].max
        initial_parameters = {}
        for name, param in self.clients[0].model.named_parameters():
            initial_parameters[name] = copy.deepcopy(param.data)

        for c in self.clients:
            assert (c.encoder.level_hvs == level_hvs).all()
            assert (c.encoder.id_hvs == id_hvs).all()
            for name, param in c.model.named_parameters():
                assert (param.data == initial_parameters[name]).all()

        self.display_info("initial state checked")



    def client_encode(self):
        for c in self.clients:
            c.encode_train()
            c.encode_test()
        self.display_info('All client data encoded')
        self.encoding_done_time = (datetime.now() - self.start_time).total_seconds()
        self.display_info("Ecoding time used: " + str(self.encoding_done_time))


    def train_clients(self):
        for c in self.clients:
            c.train(self.epoch)
        self.round_time.append((datetime.now() - self.start_time).total_seconds())


    def client_batch_input(self):
        for c in self.clients:
            c.batch_input()
        self.display_info('All client inputs batched')


    def global_eval(self):
        self.display_info("========Gloabl Eval========")
        client_f1s = []
        client_accs = []
        for c in self.clients:
            f1, acc = c.eval()
            client_f1s.append(f1)
            client_accs.append(acc)
        global_f1 = sum(client_f1s)/len(client_f1s)
        global_acc = sum(client_accs)/len(client_accs)
        self.display_info("Global eval F1: {0} - Global eval Acc: {1}".format(global_f1,global_acc))
        self.eval_results_f1.append(client_f1s)
        self.eval_results_acc.append(client_accs)



    def global_test(self):
        self.display_info("========Gloabl Test========")
        client_f1s = []
        client_accs = []
        for c in self.clients:
            f1, acc = c.test()
            client_f1s.append(f1)
            client_accs.append(acc)
        global_f1 = sum(client_f1s)/len(client_f1s)
        global_acc = sum(client_accs)/len(client_accs)
        self.display_info("Global test F1: {0} - Global test Acc: {1}".format(global_f1,global_acc))
        self.test_results_f1.append(client_f1s)
        self.test_results_acc.append(client_accs)




    def save_result(self):
        with open(self.result_path + '\\exp_config.txt', 'w') as configfile:
            self.configs.write(configfile)
        with open(self.result_path + '\\eval_f1.csv', 'w') as f:
            write = csv.writer(f)
            write.writerows(self.eval_results_f1)
        with open(self.result_path + '\\eval_acc.csv', 'w') as f:
            write = csv.writer(f)
            write.writerows(self.eval_results_acc)
        with open(self.result_path + '\\test_f1.csv', 'w') as f:
            write = csv.writer(f)
            write.writerows(self.test_results_f1)
        with open(self.result_path + '\\test_acc.csv', 'w') as f:
            write = csv.writer(f)
            write.writerows(self.test_results_acc)
        with open(self.result_path + '\\time_step.csv', 'w') as f:
            write = csv.writer(f)
            write.writerows([[self.encoding_done_time] + self.round_time])

        self.val_acc = np.array(self.eval_results_acc)
        self.test_acc = np.array(self.test_results_acc)
        self.val_f1 = np.array(self.eval_results_f1)
        self.test_f1 = np.array(self.test_results_f1)

        plt.figure(figsize=(25,15))

        for c in range(len(self.clients)):
            x = [i for i in range(self.round+1)]
            val_acc = self.val_acc[:,c] * 100
            test_acc = self.test_acc[:,c] * 100
            val_f1 = self.val_f1[:,c] * 100
            test_f1 = self.test_f1[:,c] * 100

            plt.plot(x, val_acc, label = "val_acc")
            plt.plot(x, test_acc, label = "test_acc")
            plt.plot(x, val_f1, label = "val_f1")
            plt.plot(x, test_f1, label = "test_f1")
            plt.ylim([0, 100])

            for xy in zip(x[::5], val_acc[::5]):
               plt.annotate('(%.2f, %.2f)' % xy, xy=xy)
            for xy in zip(x[::5], test_acc[::5]):
               plt.annotate('(%.2f, %.2f)' % xy, xy=xy)
            for xy in zip(x[::5], val_f1[::5]):
               plt.annotate('(%.2f, %.2f)' % xy, xy=xy)
            for xy in zip(x[::5], test_f1[::5]):
               plt.annotate('(%.2f, %.2f)' % xy, xy=xy)

            plt.title("Client " + str(c))
            plt.legend()
            plt.savefig(self.result_path + '\\' + 'Client' + str(c) + ".png")
            plt.clf()

        x_round = [i for i in range(self.round+1)]
        x_time = [self.encoding_done_time] + self.round_time
        for graph in ['Round','Time']:
            if graph == 'Round':
                x = x_round
            if graph == 'Time':
                x = x_time
            y_val_acc = (self.val_acc.sum(axis=1) / len(self.clients)) * 100
            y_test_acc = (self.test_acc.sum(axis=1) / len(self.clients)) * 100
            y_val_F1 = (self.val_f1.sum(axis=1) / len(self.clients)) * 100
            y_test_f1 = (self.test_f1.sum(axis=1) / len(self.clients)) * 100

            plt.plot(x, y_val_acc, label = "Validation Acc")
            plt.plot(x, y_test_acc, label = "Test Acc")
            plt.plot(x, y_val_F1, label = "Validation F1")
            plt.plot(x, y_test_f1, label = "Test F1")
            plt.ylim([0, 100])

            for xy in zip(x[::5], y_val_acc[::5]):
               plt.annotate('(%.2f, %.2f)' % xy, xy=xy)
            for xy in zip(x[::5], y_test_acc[::5]):
               plt.annotate('(%.2f, %.2f)' % xy, xy=xy)
            for xy in zip(x[::5], y_val_F1[::5]):
               plt.annotate('(%.2f, %.2f)' % xy, xy=xy)
            for xy in zip(x[::5], y_test_f1[::5]):
               plt.annotate('(%.2f, %.2f)' % xy, xy=xy)
            
            plt.title("All Clients")
            plt.ylabel('Acc/F1')
            plt.xlabel(graph)
            plt.legend()
            plt.savefig(self.result_path + '\\ALL_Clients_' + graph + '_.png')
            plt.clf()




###################################################################################################################################
    def aggregate_network_nofed(self):
        self.display_info('Network aggregated (NoFed)')

    def aggregate_network_qkv(self):
        n_clients = len(self.clients)
        client_parameters = {}
        for name, param in self.clients[0].model.named_parameters():
            if param.requires_grad and (name in ['query_weights.weight','value_weights.weight','key_weights.weight']):
                client_parameters[name] = []
        for c in self.clients:
            for name, param in c.model.named_parameters():
                if param.requires_grad and (name in ['query_weights.weight','value_weights.weight','key_weights.weight']):
                    client_parameters[name].append(param.data)
        for key in client_parameters.keys():
            client_parameters[key] = sum(client_parameters[key]) / n_clients
        for c in self.clients:
            for name, param in c.model.named_parameters():
                if param.requires_grad and (name in ['query_weights.weight','value_weights.weight','key_weights.weight']):
                    param.data = copy.deepcopy(client_parameters[name])
        self.display_info('Network aggregated (QKV)')


    def aggregate_network_fedper(self):
        n_clients = len(self.clients)
        client_parameters = {}
        for name, param in self.clients[0].model.named_parameters():
            if param.requires_grad and (name in ['query_weights.weight','value_weights.weight','key_weights.weight','projection.weight','projection.bias']):
                client_parameters[name] = []
        for c in self.clients:
            for name, param in c.model.named_parameters():
                if param.requires_grad and (name in ['query_weights.weight','value_weights.weight','key_weights.weight','projection.weight','projection.bias']):
                    client_parameters[name].append(param.data)
        for key in client_parameters.keys():
            client_parameters[key] = sum(client_parameters[key]) / n_clients
        for c in self.clients:
            for name, param in c.model.named_parameters():
                if param.requires_grad and (name in ['query_weights.weight','value_weights.weight','key_weights.weight','projection.weight','projection.bias']):
                    param.data = copy.deepcopy(client_parameters[name])
        self.display_info('Network aggregated (FedPer)')


    def aggregate_network_fedavg(self):
        n_clients = len(self.clients)
        client_parameters = {}
        for name, param in self.clients[0].model.named_parameters():
            if param.requires_grad:
                client_parameters[name] = []
        for c in self.clients:
            for name, param in c.model.named_parameters():
                if param.requires_grad:
                    client_parameters[name].append(param.data)
        for key in client_parameters.keys():
            client_parameters[key] = sum(client_parameters[key]) / n_clients
        for c in self.clients:
            for name, param in c.model.named_parameters():
                if param.requires_grad:
                    param.data = copy.deepcopy(client_parameters[name])
        self.display_info('Network aggregated (FedAvg)')


    def aggregate_network_proximity(self):
        clients_parameters = []
        for c in self.clients:
            parameters = {}
            for name, param in c.model.named_parameters():
                if param.requires_grad:
                    parameters[name] = param.data
            clients_parameters.append(parameters)

        softmax_score = self.cal_proximity_matrix(clients_parameters)

        self.display_info(str(softmax_score))

        for i in range(self.num_client):
            for name, param in self.clients[i].model.named_parameters():
                if param.requires_grad:
                    param_list = [p[name] for p in clients_parameters]
                    score_list = softmax_score[i].tolist()
                    param_update = sum([a*b for a,b in zip(param_list,score_list)])
                    param.data = copy.deepcopy(param_update)

        self.display_info('Network aggregated (Proximity)')



        
    def aggregate_network_attn_proximity(self):
        clients_parameters = []
        for c in self.clients:
            parameters = {}
            for name, param in c.model.named_parameters():
                if param.requires_grad and 'classifier' not in name:
                    parameters[name] = param.data
            clients_parameters.append(parameters)

        softmax_score = self.cal_proximity_matrix(clients_parameters)

        self.display_info(str(softmax_score))

        for i in range(self.num_client):
            for name, param in self.clients[i].model.named_parameters():
                if param.requires_grad and 'classifier' not in name:
                    param_list = [p[name] for p in clients_parameters]
                    score_list = softmax_score[i].tolist()
                    param_update = sum([a*b for a,b in zip(param_list,score_list)])
                    param.data = copy.deepcopy(param_update)

        self.display_info('Network aggregated (Attn Proximity)')





    def cal_proximity_matrix(self,clients_parameters):
        param_order = []
        for key in clients_parameters[0].keys():
            param_order.append(key)

        parameter_space_vector = []
        for parameters in clients_parameters:
            vector = []
            for key in param_order:
                vector.append(torch.flatten(parameters[key]))
            parameter_space_vector.append(torch.cat(vector,dim=0))

        parameter_space_vector = torch.stack(parameter_space_vector)

        cos_sim_matrix = []
        for i in range(self.num_client):
            cos_sim_matrix.append(torch.nn.functional.cosine_similarity(parameter_space_vector[i],parameter_space_vector))

        softmax_scores = []
        for i in range(self.num_client):
            softmax_scores.append(torch.nn.functional.softmax(cos_sim_matrix[i]/self.temperature,dim=0))

        return softmax_scores



###################################################################################################################################


    def start(self):
        self.init_clients()
        self.initial_state_check()

        self.client_encode()
        self.client_batch_input()
        self.global_eval()
        self.global_test()
        for r in range(self.round):
            self.display_info("====================round{0}====================".format(r))
            self.train_clients()
            self.global_eval()
            self.global_test()
            self.aggregation()
        self.save_result()



    def measure_epoch_time(self):
        self.display_info("Only measuring Epoch Time on one client with full modality")
        with open(self.datapath+'client_{0}_data'.format(str("0")), 'rb') as data_file:
            data = pickle.load(data_file)
            client = Client(self.configs,data)
        client.init()
        start_time = datetime.now()
        client.encode_train()
        client.encode_test()
        self.encoding_time = (datetime.now() - start_time).total_seconds()
        client.batch_input()
        for i in range(5):
            start_time = datetime.now()
            client.train(self.epoch)
            self.epoch_time.append((datetime.now() - start_time).total_seconds())
            self.display_info(str(self.epoch_time[-1]))

        with open(self.result_path + '\\epoch_time_measurment.csv', 'w') as f:
            write = csv.writer(f)
            write.writerows([self.epoch_time])

        average = sum(self.epoch_time) / len(self.epoch_time)
        self.display_info("Encoding Time: " + str(self.encoding_time))
        self.display_info("Average time per epoch: " + str(average))

        exit()
































