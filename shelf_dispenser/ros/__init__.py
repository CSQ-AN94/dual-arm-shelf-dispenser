"""ROS-side entry points, run by the system Python, not this one.

Each module here is invoked as a script by path from ``planner``, so
they import each other as flat siblings rather than as a package.
That is deliberate: they never share an interpreter with the caller.
"""
