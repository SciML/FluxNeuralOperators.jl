using DataDeps

export get_double_pendulum_chaotic_data

function register_double_pendulum_chaotic()
    register(DataDep(
        "DoublePendulumChaotic",
        """
        Dataset was generated on the basis of 21 individual runs of a double pendulum.
        Each of the recorded sequences lasted around 40s and consisted of around 17500 frames.

        * `x_red`: Horizontal pixel coordinate of the red point (the central pivot to the first pendulum)
        * `y_red`: Vertical pixel coordinate of the red point (the central pivot to the first pendulum)
        * `x_green`: Horizontal pixel coordinate of the green point (the first pendulum)
        * `y_green`: Vertical pixel coordinate of the green point (the first pendulum)
        * `x_blue`: Horizontal pixel coordinate of the blue point (the second pendulum)
        * `y_blue`: Vertical pixel coordinate of the blue point (the second pendulum)

        Page: https://developer.ibm.com/exchanges/data/all/double-pendulum-chaotic/
        """,
        "https://dax-cdn.cdn.appdomain.cloud/dax-double-pendulum-chaotic/2.0.1/double-pendulum-chaotic.tar.gz",
        "4ca743b4b783094693d313ebedc2e8e53cf29821ee8b20abd99f8fb4c0866f8d",
        post_fetch_method=unpack
    ))
end

function get_double_pendulum_chaotic_data()
    data_path = joinpath(datadep"DoublePendulumChaotic", "original", "dpc_dataset_csv")
end