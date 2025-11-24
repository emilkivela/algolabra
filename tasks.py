from invoke import task

@task
def start(ctx):
    ctx.run("cd src; poetry run flask run")