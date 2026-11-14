/**
 * With the server running `ZOTERO_UPLOAD`, the Upload button adds PDFs to the
 * Zotero library: the labels say so, and a successful upload must not start the
 * pipeline activity probe, because nothing was queued. Without the flag the
 * dialog is the upstream one.
 *
 * The fake backend sits at the axios adapter seam (see ClearDocumentsDialog.test)
 * so the real `uploadDocument` runs and the request it sends is what is checked.
 */
import { afterEach, beforeEach, describe, expect, mock, test } from 'bun:test'
import { cleanup, screen, waitFor } from '@testing-library/react'
import userEvent from '@testing-library/user-event'

import * as api from '@/api/lightrag'
import { useBackendState } from '@/stores/state'
import { renderWithProviders } from '@/test/render'
import UploadDocumentsDialog from './UploadDocumentsDialog'

const zoteroBody = {
  status: 'success',
  message: 'paper.pdf added to the Zotero library as item ITEM0001 (metadata from CrossRef).',
  item_key: 'ITEM0001',
  attachment_key: 'ATT00001',
  title: 'A title',
  doi: '10.1/x',
  metadata: 'crossref',
  sync_triggered: false,
  sync_message: 'No sync was triggered.',
  sync_state: 'not_configured'
}

const uploads: string[] = []
let uploadResponse: { status: number; data: unknown } = { status: 200, data: zoteroBody }

const reply = (config: any, status: number, data: unknown) => {
  const response = { data, status, statusText: '', headers: {}, config }
  if (status >= 400) {
    return Promise.reject(Object.assign(new Error(`status ${status}`), { config, response }))
  }
  return Promise.resolve(response)
}

const setZotero = (enabled: boolean | undefined) =>
  useBackendState.setState({
    status: { status: 'healthy', configuration: { zotero_upload: enabled } } as any
  })

beforeEach(() => {
  api.__setAxiosAdapterForTests(async (config: any) => {
    if (config.url === '/documents/supported_file_types') {
      return reply(config, 200, { supported_extensions: ['.pdf'], engines: {} })
    }
    if (config.url === '/documents/upload') {
      uploads.push((config.data as FormData).get('file') ? 'file' : 'nofile')
      return reply(config, uploadResponse.status, uploadResponse.data)
    }
    return reply(config, 200, {})
  })
})

afterEach(() => {
  api.__setAxiosAdapterForTests(undefined)
  cleanup()
  uploads.length = 0
  uploadResponse = { status: 200, data: zoteroBody }
  useBackendState.setState({ status: null })
})

const dropPdf = async (onUploadBatchAccepted: () => void) => {
  const user = userEvent.setup()
  const { baseElement } = renderWithProviders(
    <UploadDocumentsDialog onUploadBatchAccepted={onUploadBatchAccepted} />
  )
  await user.click(screen.getAllByRole('button')[0])
  await waitFor(() => expect(baseElement.querySelector('input[type=file]')).toBeTruthy())
  const input = baseElement.querySelector('input[type=file]') as HTMLInputElement
  await waitFor(() => expect(input.disabled).toBe(false))
  await user.upload(input, new File(['%PDF-1.4'], 'paper.pdf', { type: 'application/pdf' }))
  return baseElement
}

describe('UploadDocumentsDialog', () => {
  test('says Add to Zotero only when the server has the flag on', () => {
    setZotero(true)
    renderWithProviders(<UploadDocumentsDialog />)
    expect(screen.getAllByRole('button')[0].textContent).toContain('Add to Zotero')
    cleanup()

    setZotero(undefined)
    renderWithProviders(<UploadDocumentsDialog />)
    expect(screen.getAllByRole('button')[0].textContent).not.toContain('Zotero')
    expect(screen.getAllByRole('button')[0].textContent).toContain('Upload')
  })

  test('a Zotero upload posts the file and does not start the activity probe', async () => {
    setZotero(true)
    const accepted = mock(() => {})
    await dropPdf(accepted)
    await waitFor(() => expect(uploads).toEqual(['file']))
    await waitFor(() => expect(screen.queryAllByText(/rejected|failed/i).length).toBe(0))
    expect(accepted.mock.calls.length).toBe(0)
  })

  test('without the flag a successful upload still starts the activity probe', async () => {
    setZotero(false)
    uploadResponse = {
      status: 200,
      data: { status: 'success', message: 'queued', track_id: 'upload_1' }
    }
    const accepted = mock(() => {})
    await dropPdf(accepted)
    await waitFor(() => expect(accepted.mock.calls.length).toBe(1))
  })

  test('a Zotero duplicate shows the server detail verbatim', async () => {
    setZotero(true)
    uploadResponse = {
      status: 409,
      data: { detail: 'Already in the Zotero library: item ITEM0001 (DOI 10.1/x).' }
    }
    await dropPdf(mock(() => {}))
    await waitFor(() =>
      expect(screen.getAllByText(/Already in the Zotero library: item ITEM0001/).length).toBe(1)
    )
  })
})
