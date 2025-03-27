import torch

def save(args, model, wandb, ep=None):
    import os
    save_dir = '/10tb/dkriukov/kuramoto/iql/' 
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    if not ep == None:
        torch.save(model.state_dict(), 
                   save_dir + '_' + args.run_name + '_' + args.reward_type + '_Ep' + str(ep) + ".pth"
                   )
        wandb.save(save_dir + '_' + args.run_name + '_' + args.reward_type + '_Ep' + str(ep) + ".pth"
                   )
    else:
        torch.save(model.state_dict(), 
                   save_dir + '_' + args.run_name + '_' + args.reward_type
                   )
        wandb.save(save_dir + '_' + args.run_name + '_' + args.reward_type
                   )