import os
import paths


# Set up the model
exec(open(paths.scripts / "luhman16b_model.py").read())

# Optimization settings
lr = 1e-3
niter = 1000

# Optimize!
loss = []
best_loss = np.inf
map_soln = model.test_point
iterator = tqdm(
    pmx.optim.optimize_iterator(pmx.optim.Adam(lr=lr), niter, start=map_soln),
    total=niter,
    disable=os.getenv("CI", "false") == "true",
)
with model:
    for obj, point in iterator:
        iterator.set_description(
            "loss: {:.3e} / {:.3e}".format(obj, best_loss)
        )
        loss.append(obj)
        if obj < best_loss:
            best_loss = obj
            map_soln = point

# Save
np.savez(
    paths.data / "luhman16b_solution.npz",
    loss=loss,
    map_soln=map_soln,
)