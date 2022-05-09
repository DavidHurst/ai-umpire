if __name__ == "__main__":
    # Get filtered detections from ball detector
    detector = BallDetector(root_dir=ROOT_DIR_PATH)
    detections_IC = detector.get_filtered_ball_detections(
        vid_fname=video_fname,
        sim_id=i,
        morph_op="close",
        morph_op_iters=1,
        morph_op_se_shape=(21, 21),
        blur_kernel_size=(33, 33),
        blur_sigma=1,
        binary_thresh=120,
        disable_progbar=False,
        struc_el=cv.MORPH_RECT,
        min_ball_travel_dist=1,
        max_ball_travel_dist=70,
        min_det_area=1,
        max_det_area=30,
    )
    print(f"Detections: shape={detections_IC.shape}, noisy_gt: \n{detections_IC}")

    # Transform z detections_IC to be in the range [-0.5COURT_LENGTH, 0.5COURT_LENGTH]
    old_z = detections_IC[:, -1]
    tformed_z = transform_nums_to_range(
        old_z, [np.min(old_z), np.max(old_z)], [-HALF_COURT_LENGTH, HALF_COURT_LENGTH]
    )
    tformed_z = np.reshape(tformed_z, (tformed_z.shape[0], 1))
    smoothed_tformed_z = savgol_filter(tformed_z.squeeze(), len(tformed_z) // 3 + 1, 3)

    # plt.plot(np.linspace(0, 10, len(smoothed_tformed_z)), smoothed_tformed_z, label="Smoothed")
    # plt.plot(np.linspace(0, 10, len(smoothed_tformed_z)), tformed_z, label="Not Smoothed")
    # plt.plot(np.linspace(0, 10, len(smoothed_tformed_z)), z[1:], label="GT")
    # plt.legend()
    # plt.tight_layout()
    # plt.show()

    ball_pos_true = load_sim_ball_pos(SIM_ID, ROOT_DIR_PATH, N_FRAMES_TO_AVERAGE)
    print(f"ball pos true shape = {ball_pos_true.shape}")
    print(ball_pos_true)
