import numpy as np
# import vtk

from streamlines.io import load, save


def filter(input, output, **kwargs):

    # Load the input streamlines using the requested parameters.
    streamlines = load(input)
    streamlines.filter(**kwargs)

    # Save the streamlines to the output file.
    save(streamlines, output)

def info(input):

    # Load the input streamlines using the requested parameters.
    streamlines = load(input)

    # Print info about the streamlines.
    out = ''
    out += '\nNumber of streamlines: {}'.format(len(streamlines))
    out += '\nMean length: {:.2f}'.format(np.mean(streamlines.lengths))

    print(out)

def reorient(input, output, **kwargs):

    # Load the input streamlines using the requested parameters.
    streamlines = load(input)
    streamlines.reorient(**kwargs)

    # Save the streamlines to the output file.
    save(streamlines, output)

def smooth(input, output, **kwargs):

    # Load the input streamlines using the requested parameters.
    streamlines = load(input)
    streamlines.smooth(**kwargs)

    # Save the streamlines to the output file.
    save(streamlines, output)


def view(filename):
    """View streamlines in interactive window"""

    # Load the streamlines to be displayed.
    streamlines = load(filename)

    # Create a new vtk renderer and rendering window.
    renderer = vtk.vtkRenderer()
    renderer.SetViewport(0.0, 0.0, 1.0, 1.0)
    renderer.SetBackground(0.0, 0.0, 0.0)
    rendering_window = vtk.vtkRenderWindow()
    rendering_window.AddRenderer(renderer)

    # Allow the user to interact with the mesh.
    interactor = vtk.vtkRenderWindowInteractor()
    interactor.SetRenderWindow(rendering_window)
    interactor.SetInteractorStyle(
        vtk.vtkInteractorStyleTrackballCamera())

    # Compute the number of points.
    nb_points = 0
    for streamline in streamlines:
        nb_points += len(streamline)

    # The vtk points and lines that represent the lines to draw.
    points = vtk.vtkPoints()
    points.SetNumberOfPoints(nb_points)
    lines = vtk.vtkCellArray()

    # Copy the point and streamline data to vtk objects.
    point_count = 0
    for streamline in streamlines:
        lines.InsertNextCell(len(streamline))
        for idx, point in enumerate(streamline):
            points.SetPoint(point_count, point[0], point[1], point[2])
            lines.InsertCellPoint(point_count)
            point_count += 1

    poly_data = vtk.vtkPolyData()
    poly_data.SetPoints(points)
    poly_data.SetLines(lines)

    mapper = vtk.vtkPolyDataMapper()
    mapper.SetInputData(poly_data)

    actor = vtk.vtkActor()
    actor.SetMapper(mapper)
    actor.GetProperty().SetOpacity(0.01)
    actor.GetProperty().SetLineWidth(1)

    # Add the actor to the renderer.
    renderer.AddActor(actor)

    # Start rendering.
    rendering_window.Render()
    interactor.Start()

    # Cleanup when the user closes the window.
    rendering_window.Finalize()
    interactor.TerminateApp()
