import cv2
import time
import os
os.environ['PYOPENGL_PLATFORM'] = 'egl'

import torch
import joblib
import shutil
import colorsys
import argparse
import numpy as np
from tqdm import tqdm
from multi_person_tracker import MPT

from lib.models.vibe import VIBE_Demo
from lib.utils.renderer import Renderer
from lib.dataset.inference import Inference
from lib.data_utils.kp_utils import convert_kps
from lib.utils.pose_tracker import run_posetracker
from lib.utils.async_cam import AsyncCamera
from lib.dataset.live_inference import LiveInference

from lib.utils.demo_utils import (
	download_youtube_clip,
	smplify_runner,
	convert_crop_cam_to_orig_img,
	prepare_rendering_results,
	video_to_images,
	images_to_video,
	download_ckpt,
)


# ========= Run Temporal SMPLify if tracking method is pose ========= #
def temporal_simplify(pred_verts,pred_cam,pred_pose,pred_betas,pred_joints3d,norm_joints2d,device,args):
	if args.run_smplify and args.tracking_method == 'pose':
		norm_joints2d = np.concatenate(norm_joints2d, axis=0)
		norm_joints2d = convert_kps(norm_joints2d, src='staf', dst='spin')
		norm_joints2d = torch.from_numpy(norm_joints2d).float().to(device)
		print('pred_verts is ',pred_verts)
		# Run Temporal SMPLify
		update, new_opt_vertices, new_opt_cam, new_opt_pose, new_opt_betas, \
		new_opt_joints3d, new_opt_joint_loss, opt_joint_loss = smplify_runner(
			pred_rotmat=torch.cat(pred_pose, dim=0),
			pred_betas=torch.cat(pred_betas, dim=0),
			pred_cam=torch.cat(pred_cam, dim=0),
			j2d=norm_joints2d,
			device=device,
			batch_size=norm_joints2d.shape[0],
			pose2aa=False,
		)



		pred_verts[0][update] = new_opt_vertices[update].to(device)
		pred_cam[0][update] = new_opt_cam[update].to(device)
		pred_pose[0][update] = new_opt_pose[update].to(device)
		pred_betas[0][update] = new_opt_betas[update].to(device)
		pred_joints3d[0][update] = new_opt_joints3d[update].to(device)
		pred_verts[0] = pred_verts[0].cpu()
		pred_cam[0] = pred_cam[0].cpu()
		pred_pose[0] = pred_pose[0].cpu()
		pred_betas[0] = pred_betas[0].cpu()
		pred_joints3d[0] = pred_joints3d[0].cpu()


	elif args.run_smplify and args.tracking_method == 'bbox':
		print('[WARNING] You need to enable pose tracking to run Temporal SMPLify algorithm!')
		print('[WARNING] Continuing without running Temporal SMPLify!..')

	return pred_verts,pred_cam,pred_pose,pred_betas,pred_joints3d,norm_joints2d

# ========= Generate VIBE Results ========= #
def get_vibe_results(pred_cam,pred_verts,pred_pose,pred_betas,pred_joints3d,joints2d_lis,bbox_lis,frame_lis,orig_dim,person_id):
	pred_cam = torch.cat(pred_cam, dim=0).cpu().numpy()
	pred_verts = torch.cat(pred_verts, dim=0).cpu().numpy()
	pred_pose = torch.cat(pred_pose, dim=0).cpu().numpy()
	pred_betas = torch.cat(pred_betas, dim=0).cpu().numpy()
	pred_joints3d = torch.cat(pred_joints3d, dim=0).cpu().numpy()



	orig_height,orig_width = orig_dim

	orig_cam = convert_crop_cam_to_orig_img(
		cam=pred_cam,
		bbox=bbox_lis,
		img_width=orig_width,
		img_height=orig_height
	)


	output_dict = {
		'pred_cam': pred_cam,
		'orig_cam': orig_cam,
		'verts': pred_verts,
		'pose': pred_pose,
		'betas': pred_betas,
		'joints3d': pred_joints3d,
		'joints2d': joints2d_lis,
		'bboxes': bbox_lis,
		'frame_ids': frame_lis,
	}


	   
	vibe_results = {}
	vibe_results[person_id] = output_dict
	return vibe_results


def render(orig_dim,frame_lis,vibe_results,image_folder,output_path,num_frames,args):  
	orig_height, orig_width = orig_dim
	renderer = Renderer(resolution=(orig_width, orig_height), orig_img=True, wireframe=args.wireframe)

	output_img_folder = f'live_result_output'
	os.makedirs(output_img_folder, exist_ok=True)

	print(f'Rendering output video, writing frames to {output_img_folder}')

	# prepare results for rendering
	frame_results = prepare_rendering_results(vibe_results, num_frames)
	mesh_color = {k: colorsys.hsv_to_rgb(np.random.rand(), 0.5, 1.0) for k in vibe_results.keys()}

	image_file_names = sorted([
		os.path.join(image_folder, x)
		for x in os.listdir(image_folder)
		if x.endswith('.png') or x.endswith('.jpg')
	])


	for frame_idx in tqdm(range(len(image_file_names))):
		img_fname = image_file_names[frame_idx]
		img = cv2.imread(img_fname)

		if(args.sideview):
			side_img = np.zeros_like(img)

		for person_id, person_data in frame_results[frame_idx].items():
			frame_verts = person_data['verts']
			frame_cam = person_data['cam']

			mc = mesh_color[person_id]

			mesh_filename = None

			if args.save_obj:
				mesh_folder = os.path.join(output_path, 'meshes', f'{person_id:04d}')
				os.makedirs(mesh_folder, exist_ok=True)
				mesh_filename = os.path.join(mesh_folder, f'{frame_idx:06d}.obj')


			img = renderer.render(
				img,
				frame_verts,
				cam=frame_cam,
				color=mc,
				mesh_filename=mesh_filename,
			)

			if args.sideview:
					side_img = renderer.render(
						side_img,
						frame_verts,
						cam=frame_cam,
						color=mc,
						angle=270,
						axis=[0,1,0],
					)
		if args.sideview:
			img = np.concatenate([img, side_img], axis=1)

		cv2.imwrite(os.path.join(output_img_folder, f'{frame_idx:06d}.png'), img)

		if args.display:
			cv2.imshow('Video', img)
			if cv2.waitKey(1) & 0xFF == ord('q'):
				break
	if args.display:
		cv2.destroyAllWindows()

	# ========= Save rendered video ========= #
	save_name = f'live_vibe_result.mp4'
	save_name = os.path.join(output_path, save_name)
	print(f'Saving result video to {save_name}')
	images_to_video(img_folder=output_img_folder, output_vid_file=save_name)
	shutil.rmtree(output_img_folder)
	shutil.rmtree(image_folder)



# ========= Save recent images from webcam for person tracking========= #
def saveToDir(images):
	for i,image in enumerate(images):
		cv2.imwrite(f'./live_imgs/{(i):06d}.png',image)
		


# ========= Uniformly pick images from camera feed ========= #
def get_images_from_captures(captures,MIN_NUM_FRAMES):
	gap = (int)((len(captures))//(MIN_NUM_FRAMES-1+1e-4))
	images = []
	i,x = 0,len(captures)-1
	while(x>=0 and i<MIN_NUM_FRAMES):
		images.append(captures[x])
		x-=gap
		i+=1
	return images




def main(args):
	device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')


	SEQ_LENGTH = args.sequence_length
	MIN_NUM_FRAMES = 1  # Don't change this
	TRACKER_BATCH_SIZE = MIN_NUM_FRAMES
	images_to_eval = []
	yolo_img_size = args.yolo_img_size



	image_folder = 'live_rendered_images'
	output_path = args.output_folder
	os.makedirs(image_folder, exist_ok=True)
	os.makedirs(output_path, exist_ok=True)
	os.makedirs('live_imgs', exist_ok=True)



	model = VIBE_Demo(
		seqlen=SEQ_LENGTH,
		n_layers=2,
		hidden_size=1024,
		add_linear=True,
		use_residual=True,
		live_inference=True
	).to(device)


	pretrained_file = download_ckpt(use_3dpw=False)
	ckpt = torch.load(pretrained_file)
	ckpt = ckpt['gen_state_dict']
	model.load_state_dict(ckpt, strict=False)
	model.eval()
	print(f'Loaded pretrained weights from \"{pretrained_file}\"')


	mot = MPT(device=device,batch_size=TRACKER_BATCH_SIZE,display=False,detector_type=args.detector,output_format='dict',yolo_img_size=yolo_img_size,)

	# An asynchronous camera implementation to run cv2 camera in background while model is running
	cap = AsyncCamera(0,display=args.live_display)

	bbox_scale = 1.1

	i = 0
	bbox_lis,frame_lis,images_lis,joints2d_lis = [],[],[],[]
	pred_cam, pred_verts, pred_pose, pred_betas, pred_joints3d, norm_joints2d = [], [], [], [], [], []

	while(True):
		# If q is pressed cap.stop will turn True
		if(cap.stop):
			break


		ret,captured_frames = cap.read()
		if(not ret):
			continue
		if(len(captured_frames)<MIN_NUM_FRAMES):
			continue

		images = get_images_from_captures(captured_frames,MIN_NUM_FRAMES)

		cap.del_frame_lis()
	
		orig_height, orig_width = images[0].shape[:2]
		orig_dim = (orig_height,orig_width)


		saveToDir(images)
		if args.tracking_method == 'pose':
			images_to_video('./live_imgs', './live_imgs/pose_video.mp4')
			tracking_results = run_posetracker('live_imgs/pose_video.mp4', staf_folder=args.staf_dir, display=args.display)
		else:
			tracking_results = mot('./live_imgs')
	
		if args.live_display:
			cap.set_display_image(images[-1])

		if(len(tracking_results.keys())==0):
			print('Unable to detect any person')


		for image in images:
			images_lis.append(image)

		if len(tracking_results.keys()) != 0:

			person_id = (list)(tracking_results.keys())[0]
			print(person_id)
			frames = tracking_results[person_id]['frames']


			bboxes,joints2d = None,None

			if args.tracking_method == 'pose':
				joints2d = tracking_results[person_id]['joints2d']
				if(joints2d_lis==[]):
					joints2d_lis = joints2d
				else:
					joints2d_lis = np.vstack([joints2d_lis,joints2d])
			else:	
				bboxes = tracking_results[person_id]['bbox']
				if(bbox_lis==[]):
					bbox_lis = bboxes
				else:
					bbox_lis = np.vstack([bbox_lis,bboxes])


			for x in (1+i+frames-MIN_NUM_FRAMES):
				frame_lis.append(x)

		
			dataset = LiveInference(
				images = images_lis[-SEQ_LENGTH:],
				frames=frame_lis[-SEQ_LENGTH:],
				bboxes=bbox_lis[-SEQ_LENGTH:],
				joints2d=joints2d_lis[-SEQ_LENGTH:] if joints2d is not None else None,
				scale=bbox_scale,
			)

			bboxes = dataset.bboxes

			if args.tracking_method == 'pose':
				if(bbox_lis==[]):
					bbox_lis = bboxes
				else:
					bbox_lis = np.vstack([bbox_lis,bboxes[-1:]])

			cap.set_bounding_box(bbox_lis[-1])

			has_keypoints = True if joints2d is not None else False
			norm_joints2d = []

			with torch.no_grad():

				# A manual implementation for getting data since dataloader is slow for few inputs
				tup = [dataset.__getitem__(x) for x in range(dataset.__len__())]


				if has_keypoints:
					for j,batch in enumerate(tup):
						tup[j], nj2d = batch
						norm_joints2d.append(nj2d[:21,:].reshape(-1, 21, 3))

				for j,x in enumerate(tup):
					tup[j] = x.unsqueeze(0)

				tup = tuple(tup)
				batch = torch.cat((tup),0)
							

				batch = batch.unsqueeze(0)
				batch = batch.to(device)

				batch_size, seqlen = batch.shape[:2]

				# Send only latest image to hmr for faster inferencing
				output = model(batch[:,-1:,:,:,:])[-1]

				pred_cam.append(output['theta'][:,-MIN_NUM_FRAMES:,:3].reshape(batch_size * MIN_NUM_FRAMES, -1))
				pred_verts.append(output['verts'][:,-MIN_NUM_FRAMES:,].reshape(batch_size * MIN_NUM_FRAMES, -1, 3))
				pred_pose.append(output['theta'][:,-MIN_NUM_FRAMES:,][:,:,3:75].reshape(batch_size * MIN_NUM_FRAMES, -1))
				pred_betas.append(output['theta'][:,-MIN_NUM_FRAMES:,][:, :,75:].reshape(batch_size * MIN_NUM_FRAMES, -1))
				pred_joints3d.append(output['kp_3d'][:,-MIN_NUM_FRAMES:,].reshape(batch_size * MIN_NUM_FRAMES, -1, 3))


				del batch

			pred_verts[-MIN_NUM_FRAMES:], pred_cam[-MIN_NUM_FRAMES:], pred_pose[-MIN_NUM_FRAMES:], pred_betas[-MIN_NUM_FRAMES:], pred_joints3d[-MIN_NUM_FRAMES:], norm_joints2d[-MIN_NUM_FRAMES:] = temporal_simplify(pred_verts[-MIN_NUM_FRAMES:], pred_cam[-MIN_NUM_FRAMES:], pred_pose[-MIN_NUM_FRAMES:], pred_betas[-MIN_NUM_FRAMES:], pred_joints3d[-MIN_NUM_FRAMES:], norm_joints2d[-MIN_NUM_FRAMES:], device, args)

			get_vibe_results(pred_cam[-MIN_NUM_FRAMES:], pred_verts[-MIN_NUM_FRAMES:], pred_pose[-MIN_NUM_FRAMES:], pred_betas[-MIN_NUM_FRAMES:], pred_joints3d[-MIN_NUM_FRAMES:],joints2d_lis[-MIN_NUM_FRAMES:], bbox_lis[-MIN_NUM_FRAMES:], frame_lis[-MIN_NUM_FRAMES], orig_dim,0)


		images = []
		i = i+1

		
		if(i==args.max_frames):
			break

	del model

	vibe_results = get_vibe_results(pred_cam, pred_verts, pred_pose, pred_betas, pred_joints3d,joints2d_lis, bbox_lis, frame_lis, orig_dim,0)
	

	if not args.no_render:
		for i,image in enumerate(images_lis):
			cv2.imwrite(f'{image_folder}/{(i):06d}.jpg',image)
		print(frame_lis)
		render(orig_dim, frame_lis, vibe_results, image_folder, output_path,len(images_lis),args)

	shutil.rmtree('live_imgs')
	print('================= END =================')



if __name__ == '__main__':
	parser = argparse.ArgumentParser()


	parser.add_argument('--output_folder', type=str,required=True,
						help='output folder to write results')

	parser.add_argument('--tracking_method', type=str, default='bbox', choices=['bbox', 'pose'],
						 help='tracking method to calculate the tracklet of a subject from the input video')

	parser.add_argument('--detector', type=str, default='yolo', choices=['yolo', 'maskrcnn'],
						help='object detector to be used for bbox tracking')

	parser.add_argument('--yolo_img_size', type=int, default=416,
						help='input image size for yolo detector')


	parser.add_argument('--staf_dir', type=str, default='/home/mkocabas/developments/openposetrack',
						help='path to directory STAF pose tracking method installed.')

	parser.add_argument('--sequence_length', type=int, default=4,
						help='sequence length of VIBE Model')


	parser.add_argument('--display', action='store_true',
						 help='visualize the results after rendering is complete')

	parser.add_argument('--run_smplify', action='store_true',
						help='run smplify for refining the results, you need pose tracking to enable it')

	parser.add_argument('--no_render', action='store_true',
						help='disable final rendering of output video.')

	parser.add_argument('--wireframe', action='store_true',
						help='render all meshes as wireframes.')

	parser.add_argument('--sideview', action='store_true',
						help='render meshes from alternate viewpoint.')

	parser.add_argument('--save_obj', action='store_true',
						help='save results as .obj files.')

	parser.add_argument('--max_frames', type = int,default = -1,
						help='maximum number of frames after which inferencing will stop')

	parser.add_argument('--live_display', action='store_true',
					help='show live display with bounding box')

	args = parser.parse_args()

	main(args)