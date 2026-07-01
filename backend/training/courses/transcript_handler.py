import datetime

# import os
import traceback

import requests

# DOC_DIR_NAME = "Security+700"
# DOC_DIR = f"{os.path.dirname(__file__)}/../../data/_courses/docs/{DOC_DIR_NAME}"

HOST = "https://api.vimeo.com"
HEADERS = {"Authorization": "Bearer b016f77abaf0c032157ac6bc3b8aae38"}


def get_vtt_content(course_id, video_id):
    try:
        project = requests.get(f"{HOST}/videos/{video_id}/texttracks", headers=HEADERS)
        project.raise_for_status()
        tracks = project.json()["data"]

        print(
            f"{datetime.datetime.now().isoformat()} | Handler  | COURSE_ID: {course_id} | VIDEO_ID: {video_id}"
            f" | TextTracks: {len(tracks)} | Fetched transcript list!"
        )

        for track in tracks:
            is_active = track["active"]
            print(
                f"{datetime.datetime.now().isoformat()} | Handler  | COURSE_ID: {course_id} | VIDEO_ID: {video_id}"
                f" | IS_ACTIVE={is_active} | Fetching transcript content..."
            )
            if not is_active:
                continue

            vtt = requests.get(tracks[0]["link"])
            vtt.raise_for_status()

            print(
                f"{datetime.datetime.now().isoformat()} | Handler  | COURSE_ID: {course_id} | VIDEO_ID: {video_id}"
                f" | Transcript content fetched!"
            )

            yield vtt.content

    except Exception as error:
        print(
            f"{datetime.datetime.now().isoformat()} | Handler  | COURSE_ID: {course_id} | VIDEO_ID: {video_id}"
            f" | Error while downloading transcript: {error}\n{traceback.format_exc()}"
        )


def videos_handler(course_id, data):
    for item in data:
        try:
            video = item["video"]
            video_id = int(video["uri"].split("/")[2])
            metadata = {
                "course_id": course_id,
                "video_id": video_id,
                "name": video["name"] or "Untitled",
                "description": video["description"] or "Empty",
            }

            for content in get_vtt_content(course_id, video_id):
                yield {"text": content, "metadata": metadata}

        except Exception as error:
            print(
                f"{datetime.datetime.now().isoformat()} | Handler  | COURSE_ID: {course_id}"
                f" | Error while iterating over videos: {error}\n{traceback.format_exc()}"
            )


def get_transcript_content_and_metadata(course_id, project_id):
    print(f"{datetime.datetime.now().isoformat()} | Handler  | COURSE_ID: {course_id} | Getting videos...")

    try:
        next_url = f"/users/216035057/projects/{project_id}/items"

        while next_url:
            response = requests.get(HOST + next_url, headers=HEADERS)
            response.raise_for_status()
            response = response.json()

            page = response["page"]
            next_url = response["paging"]["next"]
            data = response["data"]

            print(
                f"{datetime.datetime.now().isoformat()} | Handler  | COURSE_ID: {course_id}"
                f" | PAGE: {page} | VIDEOS: {len(data)} | Fetched videos list!"
            )

            yield from videos_handler(course_id, data)

    except Exception as error:
        print(
            f"{datetime.datetime.now().isoformat()} | Handler  | COURSE_ID: {course_id}"
            f" | Error while fetching transcript content and metadata: {error}\n{traceback.format_exc()}"
        )
