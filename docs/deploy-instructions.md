## Sim to Sim Transfer

Before deploying a new policy on the bot, first try deploying it in the Webots simulation.

### Webots Simulator
There are quite a few dependencies for running the Webots simulator. The cleanest solution that does not require many local installs is to run it in a docker container (which already includes all dependencies) on the `robovision` server.

This docker container has webots simulator, compiled `booster_robotics_sdk`, and our `booster_deploy` repo.

Install Steps:

1. Install TurboVNC viewer locally from https://github.com/TurboVNC/turbovnc/releases (choose the installation method based your local machine).
    
    The Webots simulator requires hardware acceleration for 3D rendering, so we run it with VirtualGL to make rendering happen on the server rather than the client. We use TurboVNC to view the rendered images on the client. We therefore don't require hardware acceleration on the client machine.

    TurboVNC and VirtualGL are already installed on `robovision`.


Start ssh session:

1. ssh into robovision and map the correct ports. This ssh session needs to be kept open while the TurboVNC remote desktop is running.
    ```
    ssh -Y -L 5901:localhost:5901 <username>@robovision.csres.utexas.edu

    ```

1. Place these lines in `~/.bashrc` and `source ~/.bashrc`
    ```
    export PATH="/opt/TurboVNC/bin:$PATH"
    export VGL_DISPLAY=:1
    export DBUS_SESSION_BUS_ADDRESS=
    export XAUTHORITY=/home/<username>/.Xauthority

    ```


1. Start vncserver if one isn't running
    ```
    vncserver
    ```
    You can list running servers with
    ```
    vncserver -list
    ```
    Example output
    ```
    (base) luisamao@robovision:~$ vncserver -list

    TurboVNC sessions:

    X DISPLAY #	PROCESS ID	NOVNC PROCESS ID
    :1		24269
    ```

On client machine (local computer):
1. Start the local TurboVNC viewer (how the viewer is started depends on your local machine). The `VNC server` field should be `localhost:<display-port>`. 
    In the `vncserver -list` output above, `<display-port>` is `1` so the field should be `localhost:1`

**In the TurboVNC Remote Desktop** 

1. Pull the docker image
    TODO(Luisa): maybe build it instead?
    ```
    docker pull docker.io/llqqmm/webots_sim:latest
    ```

1. Start the docker container with 
    ```
    docker run -it \
        --device nvidia.com/gpu=all \
        -e DISPLAY=$DISPLAY \
        -v $XAUTHORITY:/tmp/.Xauthority:ro \
        -e XAUTHORITY=/tmp/.Xauthority \
        -v /tmp/.X11-unix:/tmp/.X11-unix  \
        --net=host  \
        localhost/webots_sim:latest
    ```

1. Check that visualizations and hw acceleration is working
    ```
    vglrun glxgears
    ```
    this should show turning gears and print the FPS to terminal

1. Running the simulator: Start `tmux` session with 3 windows:
    1. webots sim:
        ```
        vglrun /usr/local/webots/webots --stdout --stderr --batch --mode=realtime /workspace/webots_simulation/worlds/T1_release.wbt
        ```
    2. booster runner
        ```
        ./booster-runner-webots-full-0.0.11.run
        ```
    3. booster deploy script
        ```
        python deploy.py --config configs/walk.yaml
        ```


### Mujoco [Todo]


## Deploy on Bot [Todo]