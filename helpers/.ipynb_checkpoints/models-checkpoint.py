import torch
import torch.nn as nn
import torch.nn.functional as F


# TODO: refactor experiment code
# TODO: refactor trainer

class UnsupervisedScarf(nn.Module):
    def __init__(self, input_size, hidden_size = 256):
        '''
        scarf pretrain model: input_size->f->g->256
        pretrain-head (f): input->256->256->256->256
        projection-head (g): f(.)->256->256
        where '->' is a layer with reLU activation 
        except final layers in f and g
        '''        
        super().__init__()
        # encoder (f)
        self.encoder = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.ReLU(),
            # nn.Dropout(0.20),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            # nn.Dropout(0.20),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            # nn.Dropout(0.20),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            # nn.Dropout(0.20)
        )
        # pretraining head (g)
        # how do we confirm we are doing l2-normalizing
        ## TODO: Expt; comment out the g-part
        # in simclr they need g for finetuning the f (pretrained resnet).
        self.projector = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            # nn.Dropout(0.20),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            # # nn.Dropout(0.20),
        )

    # pre training
    def forward(self, x):
        e = self.encoder(x)
        p = self.projector(e)
        return e, p
        
        
class SupervisedScarf(nn.Module):
    def __init__(self, input_size, output_size, encoder, hidden_size = 256):
        '''
        scarf finetune model: input_size->f->h->256
        pretrained-head (f): input->256->256->256->256
        classification-head (h): f(.)->256->256
        where '->' is a layer with reLU activation 
        except final layers in f and h
        '''
        super().__init__()
        # pretrained (f)
        self.encoder = encoder
        # supervised head (h)
        self.classification_layer = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            # nn.Dropout(0.20),
            nn.Linear(hidden_size, output_size),
            # nn.ReLU(),
            # nn.Dropout(0.20),            
        )
        self.classifier = nn.Sequential(self.encoder, self.classification_layer)
        self.activation = nn.Softmax(dim=1)
        
        self.output_size = output_size
        if output_size == 1:
            self.activation = nn.Sigmoid()

    # fine-tuning
    def forward(self, x):
        intermediate = self.classifier(x)
        
        return self.activation(intermediate) if self.output_size==1 else intermediate