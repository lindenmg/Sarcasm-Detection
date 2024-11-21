from random import uniform

from tensorboardX import SummaryWriter

sessions = [
    '',
]

for session in range(3):
    writer = SummaryWriter(log_dir='runs/session_%d' % session)
    for epoch in range(1, 101):
        train_acc = uniform(0.005, 0.009) * epoch
        val_acc = uniform(0.005, 0.009) * epoch

        train_loss = uniform(50, 100) / epoch
        val_loss = uniform(50, 100) / epoch

        writer.add_scalars('lr', {'lr': 42}, epoch)
        writer.add_scalars('loss', {'train_loss': train_loss, 'val_loss': val_loss}, epoch)
        writer.add_scalars('acc', {'train_acc': train_acc, 'val_acc': val_acc}, epoch)

        writer.add_text('session_01', text_string='tfidf=100%d' % (epoch * 2), global_step=epoch)

    writer.close()
pass
