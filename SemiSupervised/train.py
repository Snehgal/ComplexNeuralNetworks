def train_mean_teacher_sassed(
    student, teacher,
    train_ann_loader, unannotated_loader,
    val_ann_loader,
    num_classes,
    n_steps_per_train_step=100,
    num_training_steps=1000,
    gamma_ramp_up_length=400,
    beta_ramp_up_length=400,
    plot_every=50,
    save_every=50,
    save_path="checkpoints/mean_teacher",
    device="cuda"
):
    import os
    import copy
    os.makedirs(save_path, exist_ok=True)

    student = student.to(device)
    teacher = teacher.to(device)
    for p in teacher.parameters():
        p.requires_grad = False

    optimizer = torch.optim.RMSprop(student.parameters(), lr=1e-4, alpha=0.9, weight_decay=1e-4)

    # Iterators for dataloaders
    ann_iter = iter(train_ann_loader)
    unann_iter = iter(unannotated_loader)

    from collections import defaultdict
    history = defaultdict(list)
    step_history = []

    global_step = 0
    ema_decay = 0.99

    while global_step < num_training_steps:
        student.train()
        for local_step in range(n_steps_per_train_step):

            # === Fetch half annotated ===
            try:
                imgs_a, masks_a, _, _ = next(ann_iter)
            except StopIteration:
                ann_iter = iter(train_ann_loader)
                imgs_a, masks_a, _, _ = next(ann_iter)

            try:
                imgs_u, _, p1_u, _ = next(unann_iter)
            except StopIteration:
                unann_iter = iter(unannotated_loader)
                imgs_u, _, p1_u, _ = next(unann_iter)

            imgs_a = imgs_a.permute(0, 3, 1, 2).to(device)
            masks_a = masks_a.to(device)
            imgs_u = imgs_u.permute(0, 3, 1, 2).to(device)
            p1_u = p1_u.to(device)

            # === Supervised Loss ===
            L_s = 0
            if masks_a is not None:
                preds_a = student(imgs_a)
                L_s = F.cross_entropy(preds_a, masks_a)

            # === Consistency Loss ===
            L_c = 0
            if imgs_u is not None:
                noisy_student = inject_noise(imgs_u)
                noisy_teacher = inject_noise(imgs_u)

                student_preds = student(noisy_student)
                with torch.no_grad():
                    teacher_preds = teacher(noisy_teacher)

                prob_s = F.softmax(student_preds, dim=1)
                prob_t = F.softmax(teacher_preds, dim=1)
                L_c = 1 - (2 * (prob_s * prob_t).sum() + 1) / (prob_s.sum() + prob_t.sum() + 1)

            # === Pseudo-supervised Loss ===
            L_u = 0
            with torch.no_grad():
                teacher_logits = teacher(imgs_u)
                teacher_preds = torch.argmax(teacher_logits, dim=1)  # (B, H, W)

            pseudo_labels = generate_pseudo_labels_via_majority_vote(
                teacher_preds, p1_u, num_classes)

            student_logits = student(imgs_u)
            pseudo_labels = torch.from_numpy(pseudo_labels).to(student_logits.device).long()
            L_u = F.cross_entropy(student_logits, pseudo_labels)


            # === Total Loss ===
            beta = np.exp(-5 * (1 - min(global_step / beta_ramp_up_length, 1)) ** 2)
            gamma = np.exp(-5 * (1 - min(global_step / gamma_ramp_up_length, 1)) ** 2)

            total_loss = L_s + beta * L_c + gamma * L_u

            optimizer.zero_grad()
            total_loss.backward()
            optimizer.step()

            # === EMA Teacher Update ===
            with torch.no_grad():
                for tp, sp in zip(teacher.parameters(), student.parameters()):
                    tp.data = ema_decay * tp.data + (1 - ema_decay) * sp.data

        global_step += 1

        

        history['supervised'].append(L_s.item())
        history['total'].append(total_loss.item())

        # === Dice on held-out annotated set ===
        def eval_model(model):
            model.eval()
            dice_scores = []
            with torch.no_grad():
             for images, masks, _, _ in val_ann_loader:
               images = images.float().to(device)     # Already batched
               masks = masks.long().to(device)
               images = images.permute(0, 3, 1, 2)     # Convert to B,C,H,W
               preds = model(images).argmax(dim=1)
               intersection = (preds == masks).sum().item()
               dice = 2 * intersection / (preds.numel() + masks.numel())
               dice_scores.append(dice)
             return np.mean(dice_scores)

        d_s = eval_model(student)
        d_t = eval_model(teacher)
        # === Logging ===
        print(f"\n[Step {global_step}] Supervised Loss: {L_s.item():.4f} | Consistency Loss: {L_c.item():.4f} | Pseudo Loss: {L_u.item():.4f} | Total: {total_loss.item():.4f}")
        print(f"\n[Step {global_step}] Student dice Loss: {d_s} | Teacher dice Loss: {d_t}")
        history['dice_student'].append(d_s)
        history['dice_teacher'].append(d_t)
        step_history.append(global_step)
        if global_step % plot_every == 0:
             plot_loss_curves(history, step_history)

        if global_step % save_every == 0:
            torch.save({
                'student': student.state_dict(),
                'teacher': teacher.state_dict(),
                'step': global_step,
                'history': history
            }, os.path.join(save_path, f"model_step_{global_step}.pt"))
            
num_classes = 9
n_out = 16
Student = UNet(n_channels=2, n_classes=num_classes, n_out_channels=n_out)
Teacher = UNet(n_channels=2, n_classes=num_classes, n_out_channels=n_out)
train_mean_teacher_sassed(
    Student, Teacher,
    train_ann_loader = train_ann_loader, unannotated_loader = unannotated_loader,
    val_ann_loader = val_ann_loader,
    num_classes = 9,
    n_steps_per_train_step=1000,
    num_training_steps=100,
    gamma_ramp_up_length = 40,
    beta_ramp_up_length=40,
    plot_every=5,
    save_every=5,
    save_path="checkpoints_mean_teacher",
    device="cuda"
)