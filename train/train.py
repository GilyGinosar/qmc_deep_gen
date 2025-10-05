import numpy as np
from tqdm import tqdm
from torch.optim import Adam
import torch
from vocalizations.qmc_deep_gen.train.losses import jacEnergy


def train_epoch(model,optimizer,loader,base_sequence,loss_function,random=True,mod=True,conditional=False):
    #device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    train_loss = 0
    epoch_losses = []
    for batch_idx,batch in enumerate(loader):
        data= batch[0]
        data = data.to(model.device)
        optimizer.zero_grad()
        if conditional:
            c = batch[1].to(torch.float32).to(model.device).view(1,-1)
            samples = model(base_sequence,random,mod,c)
        else:
            samples = model(base_sequence,random,mod)
        loss = loss_function(samples, data)
        loss.backward()
        train_loss += loss.item()
        optimizer.step()
        epoch_losses.append(loss.item())

    return epoch_losses,model,optimizer

def train_epoch_verbose(model,optimizer,loader,base_sequence,loss_function,random=True,mod=True,conditional=False):
    #device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    train_loss = 0
    epoch_losses = []
    for batch_idx, batch in tqdm(enumerate(loader),total=len(loader)):
        data = batch[0]
        data = data.to(model.device)
        optimizer.zero_grad()
        if conditional:
            c = batch[1].to(torch.float32).to(model.device).view(1,-1)
            samples = model(base_sequence,random,mod,c)
        else:
            samples = model(base_sequence,random,mod)
        loss = loss_function(samples, data)
        loss.backward()
        train_loss += loss.item()
        optimizer.step()
        epoch_losses.append(loss.item())

    return epoch_losses,model,optimizer

def test_epoch(model,loader,base_sequence,loss_function,conditional=False):

    #device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    test_loss = 0
    epoch_losses = []
    with torch.no_grad():
        for batch_idx, batch in enumerate(tqdm(loader)):
            data = batch[0]
            data = data.to(model.device)
            if conditional:
                c = batch[1].to(torch.float32).to(model.device).view(1,-1)
                samples = model(base_sequence,random=True,mod=True,c=c)
            else:
                samples = model(base_sequence,random=True,mod=True)
            #samples = model(base_sequence)
            loss = loss_function(samples, data)
            test_loss += loss.item()
            epoch_losses.append(loss.item())

    return epoch_losses

# --- put near your imports ---
import os, time, json, numpy as np, torch

def save_checkpoint(epoch, model, optimizer, losses, out_dir="runs", run_id=None, is_final=False, extra=None):
    os.makedirs(out_dir, exist_ok=True)
    run_id = run_id or time.strftime("%Y%m%d_%H%M%S")
    tag = f"final_{run_id}.pt" if is_final else f"ckpt_{run_id}_epoch{epoch:04d}.pt"
    path = os.path.join(out_dir, tag)
    tmp  = path + ".tmp"  # atomic-ish write on same volume

    state = {
        "epoch": epoch,                           # next epoch to run
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "losses": losses,                         # list of per-batch losses so far
        "run_id": run_id,
        "extra": extra or {},
        "torch_version": torch.__version__,
    }
    torch.save(state, tmp)
    os.replace(tmp, path)                        # atomic replace
    # also dump losses alone for quick plotting in anything
    np.save(os.path.join(out_dir, f"losses_{run_id}.npy"), np.asarray(losses, dtype=np.float32))
    return path

def load_checkpoint(path, model, optimizer=None, map_location="cpu"):
    state = torch.load(path, map_location=map_location)
    model.load_state_dict(state["model_state_dict"])
    if optimizer is not None and "optimizer_state_dict" in state:
        optimizer.load_state_dict(state["optimizer_state_dict"])
    return state["epoch"], state.get("losses", []), state

from torch.optim import Adam
from tqdm import tqdm

def train_loop(model, loader, base_sequence, loss_function, nEpochs=100, verbose=False,
               random=True, mod=True, conditional=False,
               out_dir="runs", run_id=None, checkpoint_every=1):

    optimizer = Adam(model.parameters(), lr=1e-3)
    losses = []
    run_id = run_id or time.strftime("%Y%m%d_%H%M%S")

    for epoch in tqdm(range(nEpochs)):
        if verbose:
            batch_loss, model, optimizer = train_epoch_verbose(
                model, optimizer, loader, base_sequence, loss_function,
                random=random, mod=mod, conditional=conditional)
        else:
            batch_loss, model, optimizer = train_epoch(
                model, optimizer, loader, base_sequence, loss_function,
                random=random, mod=mod, conditional=conditional)

        losses += batch_loss

        if verbose:
            print(f"Epoch {epoch+1} avg loss: {float(np.mean(batch_loss)):.4f}")

        # save every N epochs
        if checkpoint_every and ((epoch + 1) % checkpoint_every == 0):
            save_checkpoint(epoch=epoch+1, model=model, optimizer=optimizer,
                            losses=losses, out_dir=out_dir, run_id=run_id, is_final=False)

    # final save
    final_path = save_checkpoint(epoch=nEpochs, model=model, optimizer=optimizer,
                                 losses=losses, out_dir=out_dir, run_id=run_id, is_final=True)
    return model, optimizer, losses


# def train_loop(model,loader,base_sequence,loss_function,nEpochs=100,verbose=False,
#                random=True,mod=True,conditional=False):
#
#     optimizer = Adam(model.parameters(),lr=1e-3)
#     losses = []
#     for epoch in tqdm(range(nEpochs)):
#
#         if verbose:
#             batch_loss,model,optimizer = train_epoch_verbose(model,optimizer,loader,base_sequence,loss_function,
#                                                  random=random,mod=mod,conditional=conditional)
#         else:
#             batch_loss,model,optimizer = train_epoch(model,optimizer,loader,base_sequence,loss_function,
#                                                  random=random,mod=mod,conditional=conditional)
#
#         losses += batch_loss
#         if verbose:
#             print(f'Epoch {epoch + 1} Average loss: {np.sum(batch_loss)/len(loader.dataset):.4f}')
#
#     return model, optimizer,losses

def train_epoch_mc(model,optimizer,loader,mc_func,loss_function):
    #device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    train_loss = 0
    epoch_losses = []
    for batch_idx, (data, _) in enumerate(loader):
        base_sequence=mc_func().to(model.device)
        data = data.to(model.device)
        optimizer.zero_grad()
        samples = model(base_sequence,random=False,mod=False)
        loss = loss_function(samples, data)
        loss.backward()
        train_loss += loss.item()
        optimizer.step()
        epoch_losses.append(loss.item())

    return epoch_losses,model,optimizer

def train_loop_mc(model,loader,loss_function,mc_func,nEpochs=100,print_losses=False):

    optimizer = Adam(model.parameters(),lr=1e-3)
    losses = []
    for epoch in tqdm(range(nEpochs)):

        batch_loss,model,optimizer = train_epoch_mc(model,optimizer,loader,mc_func,loss_function)

        losses += batch_loss
        if print_losses:
            print(f'Epoch {epoch + 1} Average loss: {np.sum(batch_loss)/len(loader.dataset):.4f}')


    return model, optimizer,losses


############ TO DO: FILL IN HERE, ADD REGULARIZER STUFF ####

def train_epoch_reg(model,optimizer,loader,base_sequence,loss_function,random=True,mod=True):
    #device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    train_loss = 0
    epoch_losses = []
    for batch_idx, (data, _) in enumerate(loader):
        data = data.to(model.device)
        optimizer.zero_grad()
        samples = model(base_sequence,random,mod)
        loss = loss_function(samples, data)
        reg = jacEnergy(samples)
        l = loss + reg
        l.backward()
        train_loss += loss.item()
        optimizer.step()
        epoch_losses.append(l.item())

    return epoch_losses,model,optimizer

def train_epoch_reg_verbose(model,optimizer,loader,base_sequence,loss_function,random=True,mod=True):
    #device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    train_loss = 0
    epoch_losses = []
    for batch_idx, (data, _) in tqdm(enumerate(loader)):
        data = data.to(model.device)
        optimizer.zero_grad()
        samples = model(base_sequence,random,mod)
        loss = loss_function(samples, data)
        reg = jacEnergy(samples)
        l = loss + reg
        l.backward()
        train_loss += l.item()
        optimizer.step()
        epoch_losses.append(l.item())

    return epoch_losses,model,optimizer

def test_epoch_reg(model,loader,base_sequence,loss_function):

    #device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    test_loss = 0
    epoch_losses = []
    with torch.no_grad():
        for batch_idx, (data, _) in enumerate(tqdm(loader)):
            data = data.to(model.device)
            samples = model(base_sequence)
            loss = loss_function(samples, data)
            test_loss += loss.item()
            epoch_losses.append(loss.item())

    return epoch_losses

def train_loop_reg(model,loader,base_sequence,loss_function,nEpochs=100,verbose=False,
               random=True,mod=True):

    optimizer = Adam(model.parameters(),lr=1e-3)
    losses = []
    for epoch in tqdm(range(nEpochs)):

        if verbose:
            batch_loss,model,optimizer = train_epoch_verbose(model,optimizer,loader,base_sequence,loss_function,
                                                 random=random,mod=mod)    
        else:
            batch_loss,model,optimizer = train_epoch(model,optimizer,loader,base_sequence,loss_function,
                                                 random=random,mod=mod)

        losses += batch_loss
        if verbose:
            print(f'Epoch {epoch + 1} Average loss: {np.sum(batch_loss)/len(loader.dataset):.4f}')

    return model, optimizer,losses


        


    
