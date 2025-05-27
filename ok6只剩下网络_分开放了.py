import open3d as o3d
import open3d.visualization.gui as gui
import open3d.visualization.rendering as rendering
import numpy as np
import copy
from model import run_network


class App:
    MENU_OPEN = 1       # 菜单栏的标识符
    MENU_SHOW = 5
    MENU_QUIT = 20
    MENU_ABOUT = 21
    MENU_LOAD_PTH = 22  # 新增的菜单标识符
    MENU_RUN_NETWORK = 23  # 新增的菜单标识符

    show = True         # 布尔变量，用于控制几何体的显示
    _picked_indicates = []      # 存储选择点的信息
    _picked_points = []
    _pick_num = 0
    _label3d_list = []

    def __init__(self):
        # 初始化GUI应用程序并创建一个窗口
        gui.Application.instance.initialize()
        self.window = gui.Application.instance.create_window("Pick Points", 1200, 600)
        w = self.window
        em = w.theme.font_size

        # 左侧场景窗口
        self._scene_left = gui.SceneWidget()
        self._scene_left.scene = rendering.Open3DScene(w.renderer)
        self._scene_left.set_on_mouse(self._on_mouse_widget3d_left)

        # 中间场景窗口
        self._scene_center = gui.SceneWidget()
        self._scene_center.scene = rendering.Open3DScene(w.renderer)
        self._scene_center.set_on_mouse(self._on_mouse_widget3d_center)

        # 右侧场景窗口
        self._scene_right = gui.SceneWidget()
        self._scene_right.scene = rendering.Open3DScene(w.renderer)
        self._scene_right.set_on_mouse(self._on_mouse_widget3d_right)

        # 信息标签
        self._info = gui.Label("")
        self._info.visible = False

        # 布局回调函数
        w.set_on_layout(self._on_layout)        # 后面有_on_layout函数
        w.add_child(self._scene_left)
        w.add_child(self._scene_center)
        w.add_child(self._scene_right)
        w.add_child(self._info)

        # ---------------Menu----------------
        if gui.Application.instance.menubar is None:
            # 文件菜单栏
            file_menu = gui.Menu()
            file_menu.add_item("Open Left", App.MENU_OPEN)
            file_menu.add_item("Open Center", App.MENU_OPEN + 1)
            file_menu.add_item("Open Right", App.MENU_OPEN + 2)
            file_menu.add_separator()
            file_menu.add_item("Load PTH File", App.MENU_LOAD_PTH)  # 新增菜单项
            file_menu.add_separator()
            file_menu.add_item("Quit", App.MENU_QUIT)

            # 显示菜单栏
            show_menu = gui.Menu()
            show_menu.add_item("Show Geometry", App.MENU_SHOW)
            show_menu.set_checked(App.MENU_SHOW, True)

            # 网络菜单栏
            network_menu = gui.Menu()
            network_menu.add_item("Run Network", App.MENU_RUN_NETWORK)  # 新增菜单项

            # 帮助菜单栏
            help_menu = gui.Menu()
            help_menu.add_item("About", App.MENU_ABOUT)
            help_menu.set_enabled(App.MENU_ABOUT, False)

            # 菜单栏
            menu = gui.Menu()
            menu.add_menu("File", file_menu)
            menu.add_menu("Show", show_menu)
            menu.add_menu("Network", network_menu)
            menu.add_menu("Help", help_menu)

            gui.Application.instance.menubar = menu

            # -----注册菜单栏事件------
            w.set_on_menu_item_activated(App.MENU_OPEN, self._menu_open_left)
            w.set_on_menu_item_activated(App.MENU_OPEN + 1, self._menu_open_center)
            w.set_on_menu_item_activated(App.MENU_OPEN + 2, self._menu_open_right)
            w.set_on_menu_item_activated(App.MENU_QUIT, self._menu_quit)
            w.set_on_menu_item_activated(App.MENU_SHOW, self._menu_show)
            w.set_on_menu_item_activated(App.MENU_LOAD_PTH, self._menu_load_pth)  # 注册事件
            w.set_on_menu_item_activated(App.MENU_RUN_NETWORK, self._on_run_network)  # 注册事件

    # 处理选择 .pth 文件的菜单事件
    def _menu_load_pth(self):
        file_picker = gui.FileDialog(gui.FileDialog.OPEN, "Select PTH file...", self.window.theme)
        file_picker.add_filter('.pth', 'PTH Files (.pth)')
        file_picker.add_filter('', 'All files')
        file_picker.set_path('.')
        file_picker.set_on_cancel(self._on_cancel)
        file_picker.set_on_done(self._on_done_load_pth)
        self.window.show_dialog(file_picker)

    def _on_done_load_pth(self, filename):
        self.window.close_dialog()
        self.pth_file = filename
        print(f"Loaded PTH file: {filename}")

    # 执行网络推理
    def _on_run_network(self):
        if hasattr(self, 'pth_file') and hasattr(self, 'pcd_left'):
            # 将左侧点云传给网络，生成新点云
            generated_pcd = run_network(self.pcd_left, self.pth_file)
            self.load_generated_cloud(generated_pcd)
        else:
            print("PTH file or left point cloud not loaded!")

    # 示例网络推理函数，用户应替换为实际的网络调用


    def load_generated_cloud(self, pcd):
        pcd.paint_uniform_color([1, 0.7, 0.0])  # 设置点云颜色为黄色
        material = rendering.MaterialRecord()
        material.shader = 'defaultUnlit'
        self._scene_center.scene.clear_geometry()
        self._scene_center.scene.add_geometry('generated_point_cloud', pcd, material)
        bounds = pcd.get_axis_aligned_bounding_box()
        self._scene_center.setup_camera(60, bounds, bounds.get_center())
        self._scene_center.force_redraw()

    # 处理在左、中、右场景窗口中的鼠标事件，选择和标记点
    def _on_mouse_widget3d_left(self, event):
        if event.type == gui.MouseEvent.Type.BUTTON_DOWN and event.is_button_down(
                gui.MouseButton.LEFT) and event.is_modifier_down(gui.KeyModifier.CTRL):

            def depth_callback(depth_image):
                x = event.x - self._scene_left.frame.x
                y = event.y - self._scene_left.frame.y
                depth = np.asarray(depth_image)[y, x]

                if depth == 1.0:
                    text = ""
                else:
                    world = self._scene_left.scene.camera.unproject(x, self._scene_left.frame.height - y, depth,
                                                                    self._scene_left.frame.width, self._scene_left.frame.height)
                    text = "Left Scene: ({:.3f}, {:.3f}, {:.3f})".format(world[0], world[1], world[2])
                    idx = self._cacl_prefer_indicate_left(world)
                    true_point = np.asarray(self.pcd_left.points)[idx]
                    self._pick_num += 1
                    self._picked_indicates.append(idx)
                    self._picked_points.append(true_point)

                    print(f"Pick point #{idx} in Left Scene at ({true_point[0]}, {true_point[1]}, {true_point[2]})")

                def draw_point():
                    self._info.text = text
                    self._info.visible = (text != "")
                    self.window.set_needs_layout()

                    if depth != 1.0:
                        label3d = self._scene_left.add_3d_label(true_point, "#" + str(self._pick_num))
                        self._label3d_list.append(label3d)

                        sphere = o3d.geometry.TriangleMesh.create_sphere(0.0025)
                        sphere.paint_uniform_color([1, 0.7, 0.0])
                        sphere.translate(true_point)
                        material = rendering.MaterialRecord()
                        material.shader = 'defaultUnlit'
                        self._scene_left.scene.add_geometry("sphere" + str(self._pick_num), sphere, material)
                        self._scene_left.force_redraw()

                gui.Application.instance.post_to_main_thread(self.window, draw_point)

            self._scene_left.scene.scene.render_to_depth_image(depth_callback)
            return gui.Widget.EventCallbackResult.HANDLED
        elif event.type == gui.MouseEvent.Type.BUTTON_DOWN and event.is_button_down(
                gui.MouseButton.RIGHT) and event.is_modifier_down(gui.KeyModifier.CTRL):
            if self._pick_num > 0:
                idx = self._picked_indicates.pop()
                point = self._picked_points.pop()

                print(f"Undo pick: #{idx} in Left Scene at ({point[0]}, {point[1]}, {point[2]})")

                self._pick_num -= 1
                if self._pick_num == 0:
                    self._info.visible = False

                def draw_point():
                    if self._pick_num > 0:
                        last_point = self._picked_points[-1]
                        text = "Left Scene: ({:.3f}, {:.3f}, {:.3f})".format(last_point[0], last_point[1], last_point[2])
                        self._info.text = text
                        self._info.visible = (text != "")
                    self._scene_left.scene.remove_geometry("sphere" + str(self._pick_num + 1))
                    self._scene_left.remove_3d_label(self._label3d_list.pop())
                    self._scene_left.force_redraw()
                    self.window.set_needs_layout()

                gui.Application.instance.post_to_main_thread(self.window, draw_point)
            return gui.Widget.EventCallbackResult.HANDLED
        return gui.Widget.EventCallbackResult.IGNORED

    def _on_mouse_widget3d_center(self, event):
        if event.type == gui.MouseEvent.Type.BUTTON_DOWN and event.is_button_down(
                gui.MouseButton.LEFT) and event.is_modifier_down(gui.KeyModifier.CTRL):
            def depth_callback(depth_image):
                x = event.x - self._scene_center.frame.x
                y = event.y - self._scene_center.frame.y
                depth = np.asarray(depth_image)[y, x]

                if depth == 1.0:
                    text = ""
                else:
                    world = self._scene_center.scene.camera.unproject(x, self._scene_center.frame.height - y, depth,
                                                                      self._scene_center.frame.width, self._scene_center.frame.height)
                    text = "Center Scene: ({:.3f}, {:.3f}, {:.3f})".format(world[0], world[1], world[2])
                    print(text)

                def draw_point():
                    self._info.text = text
                    self._info.visible = (text != "")
                    self.window.set_needs_layout()

                gui.Application.instance.post_to_main_thread(self.window, draw_point)

            self._scene_center.scene.scene.render_to_depth_image(depth_callback)
            return gui.Widget.EventCallbackResult.HANDLED
        return gui.Widget.EventCallbackResult.IGNORED

    def _on_mouse_widget3d_right(self, event):
        if event.type == gui.MouseEvent.Type.BUTTON_DOWN and event.is_button_down(
                gui.MouseButton.LEFT) and event.is_modifier_down(gui.KeyModifier.CTRL):
            def depth_callback(depth_image):
                x = event.x - self._scene_right.frame.x
                y = event.y - self._scene_right.frame.y
                depth = np.asarray(depth_image)[y, x]

                if depth == 1.0:
                    text = ""
                else:
                    world = self._scene_right.scene.camera.unproject(x, self._scene_right.frame.height - y, depth,
                                                                     self._scene_right.frame.width, self._scene_right.frame.height)
                    text = "Right Scene: ({:.3f}, {:.3f}, {:.3f})".format(world[0], world[1], world[2])
                    print(text)

                def draw_point():
                    self._info.text = text
                    self._info.visible = (text != "")
                    self.window.set_needs_layout()

                gui.Application.instance.post_to_main_thread(self.window, draw_point)

            self._scene_right.scene.scene.render_to_depth_image(depth_callback)
            return gui.Widget.EventCallbackResult.HANDLED
        return gui.Widget.EventCallbackResult.IGNORED

    def _cacl_prefer_indicate_left(self, pick_point):
        if self.pcd_left:
            points = np.asarray(self.pcd_left.points)
            dist = np.sum((points - pick_point) ** 2, axis=1)
            return np.argmin(dist)
        else:
            return None

    def _menu_open_left(self):
        self._on_menu_open("Left", "pcd_left")

    def _menu_open_center(self):
        self._on_menu_open("Center", "pcd_center")

    def _menu_open_right(self):
        self._on_menu_open("Right", "pcd_right")

    def _menu_quit(self):
        gui.Application.instance.quit()

    def _menu_show(self):
        menu = gui.Application.instance.menubar.get_menu("Show")
        self.show = not menu.is_checked(App.MENU_SHOW)
        menu.set_checked(App.MENU_SHOW, self.show)
        if self.show:
            self._scene_left.scene.show_skybox(True)
        else:
            self._scene_left.scene.show_skybox(False)

    def _on_menu_open(self, id, flag):
        file_picker = gui.FileDialog(gui.FileDialog.OPEN, "Select Point Cloud...", self.window.theme)
        file_picker.add_filter('.ply .pcd .xyz', 'Point Cloud Files')
        file_picker.add_filter('', 'All files')
        file_picker.set_path('.')
        file_picker.set_on_cancel(self._on_cancel)
        file_picker.set_on_done(lambda filename: self._on_done_open(filename, id, flag))
        self.window.show_dialog(file_picker)

    def _on_done_open(self, filename, id, flag):
        self.window.close_dialog()
        pcd = o3d.io.read_point_cloud(filename)
        pcd.paint_uniform_color([1, 0.7, 0.0])  # 设置点云颜色为黄色
        material = rendering.MaterialRecord()
        material.shader = 'defaultLit'
        bounds = pcd.get_axis_aligned_bounding_box()
        if id == "Left":
            self._scene_left.scene.clear_geometry()
            self._scene_left.scene.add_geometry(flag, pcd, material)
            self._scene_left.setup_camera(60, bounds, bounds.get_center())
            self._scene_left.force_redraw()
        elif id == "Center":
            self._scene_center.scene.clear_geometry()
            self._scene_center.scene.add_geometry(flag, pcd, material)
            self._scene_center.setup_camera(60, bounds, bounds.get_center())
            self._scene_center.force_redraw()
        elif id == "Right":
            self._scene_right.scene.clear_geometry()
            self._scene_right.scene.add_geometry(flag, pcd, material)
            self._scene_right.setup_camera(60, bounds, bounds.get_center())
            self._scene_right.force_redraw()
        setattr(self, flag, pcd)

    def _on_cancel(self):
        self.window.close_dialog()

    def _on_layout(self, layout_context):
        r = self.window.content_rect
        self._scene_left.frame = gui.Rect(r.x, r.y, int(r.width / 3), r.height)
        self._scene_center.frame = gui.Rect(self._scene_left.frame.get_right(), r.y, int(r.width / 3), r.height)
        self._scene_right.frame = gui.Rect(self._scene_center.frame.get_right(), r.y, int(r.width / 3), r.height)
        self._info.frame = gui.Rect(r.x + 0.01 * r.width, r.y, r.width, 0.05 * r.height)

    def run(self):
        gui.Application.instance.run()


if __name__ == "__main__":
    app = App()
    app.run()
