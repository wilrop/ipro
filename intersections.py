def get_pairwise_intersections(box, found_boxes):
    """Get the pairwise intersections between the box and the found boxes.

    Args:
        box (Box): The box.
        found_boxes (List[Box]): The found boxes.

    Returns:
        List[Tuple(float)]: The pairwise intersections between the box and the found boxes.
    """
    intersections = []
    own_min_x, own_max_x = box.nadir[0], box.ideal[0]
    min_xs = [box.nadir[0] for box in found_boxes]
    max_xs = [box.ideal[0] for box in found_boxes]

    to_check = []
    for idx, (min_x, max_x) in enumerate(zip(min_xs, max_xs)):
        if own_min_x <= min_x <= own_max_x or (own_min_x <= max_x <= own_max_x):
            to_check.append(found_boxes[idx])

    for found_box in to_check:
        intersecting_box = box.get_intersecting_box(found_box)
        if intersecting_box is not None:
            intersections.append(intersecting_box)
    return intersections


def compute_total_vol(new_box, found_boxes, curr_vol):
    """Compute the total area covered by the found boxes. These may overlop which is taken into account.

    Note:
        This computes the formula A u B = A + B - A n B.

    Args:
        new_box (Box): The new box.
        found_boxes (List[Box]): The found boxes.
        curr_vol (float): The current volume.

    Returns:
        float: The total area covered by the found boxes.
    """
    intersecting_boxes = get_pairwise_intersections(new_box, found_boxes)
    total_vol = curr_vol + new_box.volume - sum([box.volume for box in intersecting_boxes])
    return total_vol
