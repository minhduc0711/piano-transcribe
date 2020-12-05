
import matplotlib.pyplot as plt
import torch
import numpy as np

from src.data.audio import onf_transform
from src.data.data_modules import MAPSDataModule
from sklearn.linear_model import SGDClassifier
from sklearn.multioutput import MultiOutputClassifier
from src.eval import compute_frame_metrics

def plot(sample_metrics): 
    frames = [i for i in range(len(sample_metrics))]
    precisions = [i['frame']['precision'] for i in sample_metrics]
    recalls = [i['frame']['recall'] for i in sample_metrics]    
    f1 = [i['frame']['f1'] for i in sample_metrics]    
    plt.plot(frames, precisions, 'r', label='precisions')
    plt.plot(frames, recalls, 'b', label='recalls')
    plt.plot(frames, f1, 'g', label='f1')
    plt.legend()
    plt.xlabel('Frames')
    plt.ylabel('Values')
    plt.title('Data evaluation')
    plt.show()

mapsDataModule = MAPSDataModule(batch_size=4, 
                                max_steps=5, 
                                sample_rate=16000, 
                                audio_transform=onf_transform, 
                                lazy_loading=True,
                                hop_length = 512)

# setup data 
mapsDataModule.setup()

train_loader = mapsDataModule.train_dataloader()
validate_loader = mapsDataModule.val_dataloader()
test_loader = mapsDataModule.test_dataloader()

# Classifier 
clf_log = MultiOutputClassifier(SGDClassifier(loss='log'))
clf_svc = MultiOutputClassifier(SGDClassifier(loss='hinge'))

for i_batch,batch in enumerate(train_loader):
    x_dim = batch['audio'].shape[0]
    y_dim = batch['audio'].shape[1] * batch['audio'].shape[2]
    batch_input = torch.reshape(batch['audio'], [x_dim, y_dim])
    batch_output = batch['frames'][:,2,:]
    clf_log.partial_fit(batch_input.numpy(), 
                    batch_output.numpy().astype(np.int), 
                    classes=[np.array([0,1]) for i in range(88)])
    clf_svc.partial_fit(batch_input.numpy(), 
                    batch_output.numpy().astype(np.int), 
                    classes=[np.array([0,1]) for i in range(88)])

sample_metric_logs = []
sample_metric_svc = []
for batch in validate_loader:
    x_dim = batch['audio'].shape[0]
    y_dim = batch['audio'].shape[1] * batch['audio'].shape[2]
    batch_input = torch.reshape(batch['audio'], [x_dim, y_dim])
    batch_output = batch['frames'][:,2,:]
    # logistic classify
    batch_pred_logs = clf_log.predict(batch_input.numpy())
    dict_score_logs = compute_frame_metrics(batch_pred_logs.flatten(), batch_output.numpy().astype(int).flatten())
    sample_metric_logs.append(dict_score_logs)
    # logistic classify
    batch_pred_svc = clf_svc.predict(batch_input.numpy())
    dict_score_svc = compute_frame_metrics(batch_pred_svc.flatten(), batch_output.numpy().astype(int).flatten())
    sample_metric_svc.append(dict_score_svc)

plot(sample_metric_logs)
plot(sample_metric_svc)



